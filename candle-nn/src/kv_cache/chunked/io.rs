//! I/O operations for reading and writing KV cache data.
//!
//! This module provides contiguous read/write operations:
//!
//! - [`read_contiguous`] - Read K/V data from the backing
//! - [`write_contiguous`] - Write K/V data to the backing

use std::cmp;

use super::{arena_chunks_for_format, Arena, ChunkedKvBacking};
use crate::{kv_cache::arena_table::N_PALETTE, CHUNK_SIZE};
use candle::{Result, Tensor};

impl ChunkedKvBacking {
    /// Read K/V data from the backing, dequantizing if necessary.
    ///
    /// Returns float tensors of shape (1, n_kv_head, len, head_dim).
    /// For quantized storage, data is dequantized on-the-fly.
    ///
    /// This is primarily for testing and CPU fallback - CUDA paged attention
    /// kernels read from arenas directly.
    pub fn read_contiguous(
        &self,
        batch_idx: usize,
        offset: usize,
        len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
        let chunk_size = CHUNK_SIZE;
        let device = &self.inner.device;

        // Map each requested logical token to its physical `(chunk_idx, in_blk)`,
        // honoring per-chunk window geometry. A logical sequence is NOT a flat
        // `pos / CHUNK_SIZE` grid: injected/windowed chunks carry a non-zero
        // `offset` (valid data starts mid-chunk) and a `usage` below CHUNK_SIZE,
        // so cumulative token counts — not chunk_size multiples — define the
        // block boundaries and the in-chunk slot.
        let positions: Vec<(usize, usize)> = {
            let seq = state.sequences[batch_idx].as_ref().ok_or_else(|| {
                candle::Error::Msg(format!("missing sequence allocation for batch {batch_idx}"))
            })?;
            let want_end = offset + len;
            let mut positions = Vec::with_capacity(len);
            let mut cum = 0usize;
            for blk in 0..seq.block_count() {
                let cw = seq
                    .chunk_at(blk)
                    .expect("block_count guarantees this chunk exists");
                let chunk_start = cum;
                cum += cw.usage as usize;
                let ov_start = chunk_start.max(offset);
                let ov_end = cum.min(want_end);
                if ov_start >= ov_end {
                    continue;
                }
                let base = cw.offset as usize;
                for pos in ov_start..ov_end {
                    positions.push((blk, base + (pos - chunk_start)));
                }
            }
            if positions.len() != len {
                candle::bail!(
                    "read_contiguous: logical range [{offset}, {want_end}) not fully populated \
                     for batch {batch_idx} (covered {} of {len} tokens)",
                    positions.len(),
                );
            }
            positions
        };

        // Gather K/V for each token in range using closure-based arena access
        self.inner.storage.read(|arena_state| {
            let arenas = arena_state.arenas();
            let mut k_slices = Vec::with_capacity(len);
            let mut v_slices = Vec::with_capacity(len);

            for tok in 0..len {
                let (blk, in_blk) = positions[tok];

                let cw = state.sequences[batch_idx]
                    .as_ref()
                    .and_then(|s| s.chunk_at(blk))
                    .ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "missing chunk allocation for batch {batch_idx} blk {blk}"
                        ))
                    })?;

                // Palette4 path: each head owns N_PALETTE K/V sub-chunks, each storing
                // CHUNK_SIZE × (head_dim / N_PALETTE) elements.
                {
                    let sub_head_dim = (head_dim / N_PALETTE).max(1);
                    let mut k_head_slices = Vec::with_capacity(n_kv_head);
                    let mut v_head_slices = Vec::with_capacity(n_kv_head);
                    for h in 0..n_kv_head {
                        let mut k_pal_slices = Vec::with_capacity(N_PALETTE);
                        let mut v_pal_slices = Vec::with_capacity(N_PALETTE);
                        for p in 0..N_PALETTE {
                            let k_gid = cw.gids.k_gid_pal(h, p);
                            let v_gid = cw.gids.v_gid_pal(h, p);

                            let k_ai = k_gid.arena_idx();
                            let k_ci = k_gid.chunk_idx();
                            let k_arena = arenas.get(&k_ai).ok_or_else(|| {
                                candle::Error::Msg(format!("arena {} not found", k_ai))
                            })?;
                            let k_data = match k_arena {
                                Arena::Float { data, .. } => data.clone(),
                                Arena::Quantized { data, format, .. } => {
                                    let kv_full = data.dequantize(device)?;
                                    let arena_chunks = arena_chunks_for_format(
                                        crate::kv_cache::KvFormat::Quantized(*format),
                                    );
                                    kv_full.reshape((arena_chunks, chunk_size, sub_head_dim))?
                                }
                            };
                            let k_slice = k_data
                                .narrow(0, k_ci, 1)?
                                .narrow(1, in_blk, 1)?
                                .unsqueeze(1)?;
                            k_pal_slices.push(k_slice);

                            let v_ai = v_gid.arena_idx();
                            let v_ci = v_gid.chunk_idx();
                            let v_arena = arenas.get(&v_ai).ok_or_else(|| {
                                candle::Error::Msg(format!("arena {} not found", v_ai))
                            })?;
                            let v_data = match v_arena {
                                Arena::Float { data, .. } => data.clone(),
                                Arena::Quantized { data, format, .. } => {
                                    let kv_full = data.dequantize(device)?;
                                    let arena_chunks = arena_chunks_for_format(
                                        crate::kv_cache::KvFormat::Quantized(*format),
                                    );
                                    kv_full.reshape((arena_chunks, chunk_size, sub_head_dim))?
                                }
                            };
                            let v_slice = v_data
                                .narrow(0, v_ci, 1)?
                                .narrow(1, in_blk, 1)?
                                .unsqueeze(1)?;
                            v_pal_slices.push(v_slice);
                        }
                        k_head_slices.push(Tensor::cat(&k_pal_slices, 3)?);
                        v_head_slices.push(Tensor::cat(&v_pal_slices, 3)?);
                    }
                    k_slices.push(Tensor::cat(&k_head_slices, 1)?);
                    v_slices.push(Tensor::cat(&v_head_slices, 1)?);
                }
            }

            // Concatenate along sequence dimension (dim 2)
            let k_out = Tensor::cat(&k_slices, 2)?;
            let v_out = Tensor::cat(&v_slices, 2)?;

            Ok((k_out, v_out))
        })?
    }

    /// Write contiguous K/V tokens into the chunked backing starting at `offset`.
    ///
    /// Expects `k` and `v` shaped like `(1, n_kv_head, len, head_dim)`.
    ///
    /// If any blocks in the write range are shared (via `share_prefix`), they will
    /// be automatically copied (COW) before writing to avoid affecting other sequences.
    pub fn write_contiguous(
        &self,
        batch_idx: usize,
        offset: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        let (b, h, len, d) = k.dims4()?;
        if b != 1 {
            candle::bail!("write_contiguous expects batch dim 1, got {b}")
        }
        if h != self.inner.n_kv_head {
            candle::bail!(
                "write_contiguous head mismatch (got {h}, expected {})",
                self.inner.n_kv_head
            )
        }
        if d != self.inner.head_dim {
            candle::bail!(
                "write_contiguous head_dim mismatch (got {d}, expected {})",
                self.inner.head_dim
            )
        }
        let (vb, vh, vlen, vd) = v.dims4()?;
        if (vb, vh, vlen, vd) != (b, h, len, d) {
            candle::bail!("write_contiguous K/V shape mismatch")
        }
        if len == 0 {
            return Ok(());
        }

        // Arenas are always created as float (even for quantized storage configs).
        // Quantization only happens via reconcile_sealed after chunks are complete.
        // Always use the float write path.
        self.write_contiguous_float(batch_idx, offset, k, v)
    }

    /// Float write path for write_contiguous.
    pub(crate) fn write_contiguous_float(
        &self,
        batch_idx: usize,
        offset: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        let (_, _, len, _) = k.dims4()?;

        // For quantized configs, arenas are created as F16 float (matching dequantize_f16).
        // For float configs, use the configured dtype.
        let storage_dtype = self.inner.storage.dtype().unwrap_or(candle::DType::F16);

        let k = if k.dtype() != storage_dtype {
            k.to_dtype(storage_dtype)?
        } else {
            k.clone()
        };
        let v = if v.dtype() != storage_dtype {
            v.to_dtype(storage_dtype)?
        } else {
            v.clone()
        };

        self.ensure_for_offset(batch_idx, offset, len)?;

        // Determine block range we'll write to
        let start_block = offset / CHUNK_SIZE;
        let end_pos = offset.saturating_add(len).saturating_sub(1);
        let end_block = (end_pos / CHUNK_SIZE) + 1;

        // Acquire state lock first
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        // Then acquire storage write lock and do all work inside
        let n_kv_head = self.inner.n_kv_head;
        let chunk_size = CHUNK_SIZE;
        let _device = &self.inner.device;

        self.inner.storage.write(|arena_state| {
            // COW any shared blocks in the write range
            let cow_occurred =
                self.ensure_blocks_writable_locked(&mut state, batch_idx, start_block, end_block)?;

            // Only sync block table to GPU if COW actually occurred
            if cow_occurred {}

            // Now write the data (still holding locks)
            let mut remaining = len;
            let mut pos = offset;
            while remaining > 0 {
                let blk = pos / chunk_size;
                let in_blk = pos % chunk_size;
                let seg = cmp::min(chunk_size - in_blk, remaining);
                let src_pos = pos - offset;

                let cw = state.sequences[batch_idx]
                    .as_ref()
                    .and_then(|s| s.chunk_at(blk))
                    .ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "missing chunk allocation for batch {batch_idx} blk {blk}"
                        ))
                    })?;

                let k_seg = k.narrow(2, src_pos, seg)?.contiguous()?;
                let v_seg = v.narrow(2, src_pos, seg)?.contiguous()?;

                // Palette4 path: each head has N_PALETTE K/V sub-chunks of width
                // head_dim / N_PALETTE. Quantized K arenas (e.g. active R16) are
                // written via QTensor::quantize_into at the proper flat element offset.
                {
                    let arenas = arena_state.arenas();
                    let sub_head_dim = (self.inner.head_dim / N_PALETTE).max(1);
                    for h in 0..n_kv_head {
                        let k_head = k_seg.narrow(1, h, 1)?.squeeze(1)?; // (1, seg, head_dim)
                        let v_head = v_seg.narrow(1, h, 1)?.squeeze(1)?; // (1, seg, head_dim)

                        for p in 0..N_PALETTE {
                            let d_start = p * sub_head_dim;
                            let k_band = k_head.narrow(2, d_start, sub_head_dim)?.contiguous()?;
                            let v_band = v_head.narrow(2, d_start, sub_head_dim)?.contiguous()?;

                            let k_gid = cw.gids.k_gid_pal(h, p);
                            let v_gid = cw.gids.v_gid_pal(h, p);

                            let k_ai = k_gid.arena_idx();
                            let k_ci = k_gid.chunk_idx();
                            let k_arena = arenas.get(&k_ai).ok_or_else(|| {
                                candle::Error::Msg(format!("arena {} not found", k_ai))
                            })?;
                            match k_arena {
                                Arena::Float { data, .. } => {
                                    let k_target = data.narrow(0, k_ci, 1)?;
                                    k_target.slice_set(&k_band, 1, in_blk)?;
                                }
                                #[cfg(feature = "cuda")]
                                Arena::Quantized { data, .. } => {
                                    let elem_offset =
                                        k_ci * chunk_size * sub_head_dim + in_blk * sub_head_dim;
                                    let mut qt = data.clone();
                                    qt.quantize_into(&k_band.flatten_all()?, elem_offset)?;
                                }
                                #[cfg(not(feature = "cuda"))]
                                Arena::Quantized { .. } => {
                                    candle::bail!("quantized chunk writes require the cuda feature")
                                }
                            }

                            if self
                                .inner
                                .single_latent
                                .load(std::sync::atomic::Ordering::Relaxed)
                            {
                                // K≡V: the V band aliases the K band just
                                // written — nothing separate to store.
                                continue;
                            }

                            let v_ai = v_gid.arena_idx();
                            let v_ci = v_gid.chunk_idx();
                            let v_arena = arenas.get(&v_ai).ok_or_else(|| {
                                candle::Error::Msg(format!("arena {} not found", v_ai))
                            })?;
                            match v_arena {
                                Arena::Float { data, .. } => {
                                    let v_target = data.narrow(0, v_ci, 1)?;
                                    v_target.slice_set(&v_band, 1, in_blk)?;
                                }
                                #[cfg(feature = "cuda")]
                                Arena::Quantized { data, .. } => {
                                    let elem_offset =
                                        v_ci * chunk_size * sub_head_dim + in_blk * sub_head_dim;
                                    let mut qt = data.clone();
                                    qt.quantize_into(&v_band.flatten_all()?, elem_offset)?;
                                }
                                #[cfg(not(feature = "cuda"))]
                                Arena::Quantized { .. } => {
                                    candle::bail!("quantized chunk writes require the cuda feature")
                                }
                            }
                        }
                    }
                }

                pos += seg;
                remaining -= seg;
            }

            Ok(())
        })?
    }

    /// Read raw quantized bytes for one complete chunk from a sealed (quantized) arena.
    ///
    /// Used during Hot→Warm eviction to extract the exact bytes already stored by
    /// the attention kernel, without any format conversion.  The returned bytes are
    /// in the token-oriented Q8_0 layout that `reconcile` produces — identical to
    /// what the kernel reads when attending to sealed blocks.
    ///
    /// # Errors
    /// Returns an error if the chunk at `block_idx` is still in a float arena.
    pub fn read_raw_sealed_chunk(
        &self,
        batch_idx: usize,
        block_idx: usize,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let n_kv_head = self.inner.n_kv_head;
        let chunk_size = CHUNK_SIZE;
        let head_dim = self.inner.head_dim;

        let cw = state.sequences[batch_idx]
            .as_ref()
            .and_then(|s| s.chunk_at(block_idx))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "read_raw_sealed_chunk: no chunk allocated for batch {} block {}",
                    batch_idx, block_idx
                ))
            })?;
        // Flat arena: each chunk = chunk_size * head_dim (one head, one side)
        let elems_per_head = chunk_size * head_dim;

        self.inner.storage.read(|arena_state| {
            let arenas = arena_state.arenas();
            let mut k_bytes = Vec::new();
            let mut v_bytes = Vec::new();

            for h in 0..n_kv_head {
                let k_gid = cw.gids.k_gid(h);
                let v_gid = cw.gids.v_gid(h);

                // Read K bytes for head h
                let k_arena = arenas.get(&k_gid.arena_idx()).ok_or_else(|| {
                    candle::Error::Msg(format!("arena {} not found", k_gid.arena_idx()))
                })?;
                match k_arena {
                    Arena::Quantized { data, .. } => {
                        let ggml_dtype = data.dtype();
                        let blocks_per_head = elems_per_head / ggml_dtype.block_size();
                        let bytes_per_head = blocks_per_head * ggml_dtype.type_size();
                        // Flat: chunk k_ci stores exactly one head's data
                        let k_offset = k_gid.chunk_idx() * bytes_per_head;
                        let raw = data.data_range(k_offset..k_offset + bytes_per_head)?;
                        k_bytes.extend_from_slice(&raw);
                    }
                    Arena::Float { .. } => {
                        candle::bail!(
                            "read_raw_sealed_chunk: head {} K is in a float arena \
                             (not yet reconciled); call reconcile before evicting",
                            h
                        )
                    }
                }

                // Read V bytes for head h
                let v_arena = arenas.get(&v_gid.arena_idx()).ok_or_else(|| {
                    candle::Error::Msg(format!("arena {} not found", v_gid.arena_idx()))
                })?;
                match v_arena {
                    Arena::Quantized { data, .. } => {
                        let ggml_dtype = data.dtype();
                        let blocks_per_head = elems_per_head / ggml_dtype.block_size();
                        let bytes_per_head = blocks_per_head * ggml_dtype.type_size();
                        // Flat: chunk v_ci stores exactly one head's data
                        let v_offset = v_gid.chunk_idx() * bytes_per_head;
                        let raw = data.data_range(v_offset..v_offset + bytes_per_head)?;
                        v_bytes.extend_from_slice(&raw);
                    }
                    Arena::Float { .. } => {
                        candle::bail!(
                            "read_raw_sealed_chunk: head {} V is in a float arena \
                             (not yet reconciled); call reconcile before evicting",
                            h
                        )
                    }
                }
            }

            Ok((k_bytes, v_bytes))
        })?
    }

    /// Read one sealed chunk's KV data into flat f32 vecs for CPU sampling.
    ///
    /// Only works when both K and V arenas for the slot are Float (not yet
    /// quantized); returns an error if any head is in a Quantized arena.
    ///
    /// Output layout for both `k_out` and `v_out`:
    ///   `[n_kv_head][head_dim][chunk_size]`
    ///
    /// The arena stores each palette sub-band as `(arena_chunks, chunk_size, sub_head_dim)`.
    /// This function reassembles them into contiguous head_dim order and transposes
    /// from `[t][pd]` → `[pd][t]` = `[d][t]`, matching what
    /// `sample_error_surface_cpu` expects.
    ///
    /// `block_idx` is the logical block index within the sequence at `batch_idx`
    /// (typically 0 for single-block batch windows used in the sampler).
    pub fn read_f32_sampler_chunk(
        &self,
        batch_idx: usize,
        block_idx: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
        let chunk_size = CHUNK_SIZE;
        let sub_head_dim = head_dim / N_PALETTE;

        // Snapshot (k_arena_idx, k_chunk_idx, v_arena_idx, v_chunk_idx) for every
        // (head, palette) pair while holding the state lock.
        let head_gids: Vec<(usize, usize, usize, usize)> = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let slot = state
                .sequences
                .get(batch_idx)
                .and_then(|s| s.as_ref())
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "read_f32_sampler_chunk: no sequence at batch {batch_idx}"
                    ))
                })?;
            let cw = slot.chunks_slice().get(block_idx).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "read_f32_sampler_chunk: no block {block_idx} for batch {batch_idx}"
                ))
            })?;
            (0..n_kv_head)
                .flat_map(|h| {
                    (0..N_PALETTE).map(move |p| {
                        let kg = cw.gids.k_gid_pal(h, p);
                        let vg = cw.gids.v_gid_pal(h, p);
                        (
                            kg.arena_idx(),
                            kg.chunk_idx(),
                            vg.arena_idx(),
                            vg.chunk_idx(),
                        )
                    })
                })
                .collect()
        };

        // Read raw band data from Float arenas under the storage lock.
        // Arena shape per sub-band: (arena_chunks, chunk_size, sub_head_dim).
        // After narrow + squeeze: (chunk_size, sub_head_dim), i.e. [t][pd] order.
        let bands: Vec<(Vec<f32>, Vec<f32>)> = self.inner.storage.read(|s| {
            let arenas = s.arenas();
            let mut bands = Vec::with_capacity(head_gids.len());
            for (idx, &(k_ai, k_ci, v_ai, v_ci)) in head_gids.iter().enumerate() {
                let _h = idx / N_PALETTE;
                let _p = idx % N_PALETTE;

                // Quant/R16 arenas store each chunk DIM-major: block `pd` holds
                // 32 tokens of dim `pd`, so the dequant flat is `[pd][t]`. The
                // interleave below (line `k_flat[t*sub_head_dim+pd]`) expects
                // token-major `[t][pd]`, so transpose the quant read. Float
                // arenas are already `[t][pd]` and pass through untouched.
                let k_flat = match arenas.get(&k_ai) {
                    None => candle::bail!("arena {k_ai} not found"),
                    Some(Arena::Quantized { data, .. }) => {
                        let arena_chunks = data.shape().elem_count() / (chunk_size * sub_head_dim);
                        data.dequantize(&candle::Device::Cpu)?
                            .reshape((arena_chunks, sub_head_dim, chunk_size))?
                            .narrow(0, k_ci, 1)?
                            .squeeze(0)?
                            .transpose(0, 1)?
                            .contiguous()?
                            .flatten_all()?
                            .to_vec1::<f32>()?
                    }
                    Some(Arena::Float { data, .. }) => data
                        .narrow(0, k_ci, 1)?
                        .squeeze(0)?
                        .to_dtype(candle::DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                };

                let v_flat = match arenas.get(&v_ai) {
                    None => candle::bail!("arena {v_ai} not found"),
                    Some(Arena::Quantized { data, .. }) => {
                        let arena_chunks = data.shape().elem_count() / (chunk_size * sub_head_dim);
                        data.dequantize(&candle::Device::Cpu)?
                            .reshape((arena_chunks, sub_head_dim, chunk_size))?
                            .narrow(0, v_ci, 1)?
                            .squeeze(0)?
                            .transpose(0, 1)?
                            .contiguous()?
                            .flatten_all()?
                            .to_vec1::<f32>()?
                    }
                    Some(Arena::Float { data, .. }) => data
                        .narrow(0, v_ci, 1)?
                        .squeeze(0)?
                        .to_dtype(candle::DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                };

                bands.push((k_flat, v_flat));
            }
            Ok(bands)
        })??;

        // Transpose each sub-band from [t][pd] → [pd][t] and interleave into
        // the full head_dim layout: out[(h * head_dim + d) * chunk_size + t].
        let mut k_out = vec![0.0f32; n_kv_head * head_dim * chunk_size];
        let mut v_out = vec![0.0f32; n_kv_head * head_dim * chunk_size];
        for (idx, (k_flat, v_flat)) in bands.into_iter().enumerate() {
            let h = idx / N_PALETTE;
            let p = idx % N_PALETTE;
            for pd in 0..sub_head_dim {
                let d = p * sub_head_dim + pd;
                for t in 0..chunk_size {
                    k_out[(h * head_dim + d) * chunk_size + t] = k_flat[t * sub_head_dim + pd];
                    v_out[(h * head_dim + d) * chunk_size + t] = v_flat[t * sub_head_dim + pd];
                }
            }
        }

        Ok((k_out, v_out))
    }
}
