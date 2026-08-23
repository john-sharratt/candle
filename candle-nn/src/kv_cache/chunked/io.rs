//! I/O operations for reading and writing KV cache data.
//!
//! This module provides contiguous read/write operations:
//!
//! - [`read_contiguous`] - Read K/V data from the backing
//! - [`write_contiguous`] - Write K/V data to the backing

use std::cmp;

use ahash::AHashMap;

use super::gid_pool::ChunkGid;
use super::head_gids::GIDS_PER_HEAD;
use super::{Arena, ChunkedKvBacking};
use crate::kv_cache::arena_table::{ArenaFormatTag, N_PALETTE};
use crate::kv_cache::KvFormat;
use crate::CHUNK_SIZE;
use candle::quantized::ggml_file::qtensor_from_ggml;
use candle::{Device, LiveTensor, Result, Tensor};

/// Read one band's whole chunk slot as floats, shaped `(chunk_size, sub_head_dim)`.
///
/// The band's tag decides how: a float tag is a direct typed view of the slot,
/// a quantized tag is dequantized from the slot's raw bytes. Only the chunk
/// knows which — the arena is a run of untyped byte slots.
pub(super) fn read_band_chunk<'a>(
    arenas: &'a AHashMap<usize, Arena>,
    gid: &ChunkGid,
    tag: ArenaFormatTag,
    chunk_size: usize,
    sub_head_dim: usize,
    device: &Device,
) -> Result<LiveTensor<'a>> {
    let ai = gid.arena_idx();
    let arena = arenas
        .get(&ai)
        .ok_or_else(|| candle::Error::Msg(format!("arena {ai} not found")))?;
    let elems = chunk_size * sub_head_dim;
    match tag.to_kv_format() {
        Some(KvFormat::Float(dtype)) => {
            arena.read_slot_typed(gid.chunk_idx(), dtype, (chunk_size, sub_head_dim))
        }
        Some(KvFormat::Quantized(qf)) => {
            let ggml = qf.to_ggml_dtype();
            let bytes = arena
                .slot_bytes(
                    gid.chunk_idx(),
                    (elems / ggml.block_size()) * ggml.type_size(),
                )?
                .to_vec1::<u8>()?;
            let qt = qtensor_from_ggml(ggml, &bytes, vec![elems], device)?;
            qt.dequantize(device)?.reshape((chunk_size, sub_head_dim))
        }
        None => candle::bail!("band tag {tag:?} names no storage format, so it cannot be read"),
    }
}

/// Write `band` (shaped `(1, seg, sub_head_dim)`) into one band's chunk slot at
/// `elem_offset` elements from the slot head.
///
/// Quantized bands go through `quantize_into` over a `QTensor` view of the
/// slot; float bands are a direct typed write.
fn write_band_chunk(
    arenas: &mut AHashMap<usize, Arena>,
    gid: &ChunkGid,
    tag: ArenaFormatTag,
    elems: usize,
    elem_offset: usize,
    band: &candle::LiveTensor<'_>,
) -> Result<()> {
    let ai = gid.arena_idx();
    let arena = arenas
        .get_mut(&ai)
        .ok_or_else(|| candle::Error::Msg(format!("arena {ai} not found")))?;
    match tag.to_kv_format() {
        Some(KvFormat::Float(_)) => arena.write_slot_typed(gid.chunk_idx(), elem_offset, band),
        Some(KvFormat::Quantized(qf)) => arena.quantize_into_slot(
            gid.chunk_idx(),
            qf,
            elems,
            elem_offset,
            &band.flatten_all()?,
        ),
        None => candle::bail!("band tag {tag:?} names no storage format, so it cannot be written"),
    }
}

impl ChunkedKvBacking {
    /// A single strided migration plan covering a whole `write_contiguous_float`,
    /// or `None` when some band needs more than a copy.
    ///
    /// The destination of band `(blk, h, p)` is a contiguous run inside that
    /// band's arena slot; the source is that band's columns of `k`/`v`, which
    /// are `head_dim / n_palette` contiguous elements per token, `head_dim`
    /// apart. One strided record each, so a layer's write is one launch instead
    /// of a walk over block × head × band.
    ///
    /// Refuses — returning `None` for the caller's walk to handle — when a band
    /// is quantized (`quantize_into_slot` computes scales; it is not a copy),
    /// when a band's storage dtype differs from the source (the walk casts per
    /// band), or when an arena is not on the GPU. Those are the cases the plan
    /// cannot express, and getting them wrong would write plausible bytes.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn try_plan_batched_write(
        &self,
        state: &super::types::BlockTableState,
        arena_state: &mut super::arena::ArenaStorageState,
        batch_idx: usize,
        offset: usize,
        len: usize,
        k: &candle::LiveTensor<'_>,
        v: &candle::LiveTensor<'_>,
        single_latent: bool,
    ) -> Result<Option<super::migrate::MigrationPlan>> {
        use candle::backend::BackendStorage;

        let dtype = k.dtype();
        if v.dtype() != dtype || !k.is_contiguous() || !v.is_contiguous() {
            return Ok(None);
        }
        let elem = dtype.size_in_bytes();
        let head_dim = self.inner.head_dim;
        let np = self.inner.n_palette();
        let sub = (head_dim / np).max(1);
        if sub * np != head_dim {
            return Ok(None);
        }

        // Base addresses of the source tensors. `[1, n_kv_head, len, head_dim]`
        // contiguous, so token `t` of head `h` starts at `(h·len + t)·head_dim`.
        // Element-type-agnostic: `CudaStorageSlice::device_ptr` hands back the
        // first element's address for any dtype, and the layout's element
        // offset scales by the dtype width — so every float dtype the arena
        // legitimately stores takes this fast path instead of a hand-rolled
        // per-dtype match silently dropping back to the per-band walk.
        let base_ptr = |t: &candle::LiveTensor<'_>| -> Option<i64> {
            let (storage, layout) = t.storage_and_layout();
            let cuda = match &*storage {
                candle::Storage::Cuda(c) => c,
                _ => return None,
            };
            let stream = cuda.device().cuda_stream();
            let base = cuda.slice.device_ptr(&stream);
            Some(base as i64 + (layout.start_offset() * elem) as i64)
        };
        let (Some(k_base), Some(v_base)) = (base_ptr(k), base_ptr(v)) else {
            return Ok(None);
        };

        let mut plan = super::migrate::MigrationPlan::new();
        let row_bytes = (sub * elem) as i64;
        let src_stride = (head_dim * elem) as i64;
        let dst_stride = row_bytes;

        let mut remaining = len;
        let mut pos = offset;
        while remaining > 0 {
            let blk = pos / CHUNK_SIZE;
            let in_blk = pos % CHUNK_SIZE;
            let seg = cmp::min(CHUNK_SIZE - in_blk, remaining);
            let src_pos = pos - offset;

            let Some(cw) = state.sequences[batch_idx]
                .as_ref()
                .and_then(|s| s.chunk_at(blk))
            else {
                return Ok(None);
            };
            let arenas = arena_state.arenas_mut();

            // Bands arrive in `(head, palette, k|v)` order — the same
            // `(h·np + p)·2 + which` flattening the walk uses — so the index
            // decomposes back to `(h, p, which)` with no per-block collection.
            for (i, (gid, tag)) in cw.bands().enumerate() {
                let which = i & 1;
                let hp = i >> 1;
                let (h, p) = (hp / np, hp % np);
                if which == 1 && single_latent {
                    // K≡V: the V bands are never stored.
                    continue;
                }
                let base_addr = if which == 0 { k_base } else { v_base };
                // Only a plain float band of the source's own dtype is a
                // copy; anything else the walk must handle.
                match tag.to_kv_format() {
                    Some(KvFormat::Float(dt)) if dt == dtype => {}
                    _ => return Ok(None),
                }
                let Some(arena) = arenas.get(&gid.arena_idx()) else {
                    return Ok(None);
                };
                let Some(slot) = arena.slot_ptr(gid.chunk_idx()) else {
                    return Ok(None);
                };
                let dst = slot as i64 + (in_blk * sub * elem) as i64;
                let src =
                    base_addr + (((h * len + src_pos) * head_dim + p * sub) * elem) as i64;
                plan.push_strided(src, dst, row_bytes, seg as i64, src_stride, dst_stride);
            }

            pos += seg;
            remaining -= seg;
        }
        Ok(Some(plan))
    }

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

            for &(blk, in_blk) in positions.iter() {
                let cw = state.sequences[batch_idx]
                    .as_ref()
                    .and_then(|s| s.chunk_at(blk))
                    .ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "missing chunk allocation for batch {batch_idx} blk {blk}"
                        ))
                    })?;

                // Band path: each head owns `n_palette` K/V sub-chunks, each
                // storing CHUNK_SIZE × (head_dim / n_palette) elements (4 for
                // GQA, LATENT_N_BANDS for the single latent).
                {
                    let np = self.inner.n_palette();
                    let sub_head_dim = (head_dim / np).max(1);
                    // The single latent splits its bands across two arenas of
                    // different dtypes (nope FP8 ‖ rope BF16); `Tensor::cat`
                    // needs a uniform dtype, so promote every band to F32 before
                    // concatenating. Uniform-format (GQA) reads keep their
                    // native dtype (cast only when single latent).
                    let single_latent = self
                        .inner
                        .single_latent
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let mut k_head_slices = Vec::with_capacity(n_kv_head);
                    let mut v_head_slices = Vec::with_capacity(n_kv_head);
                    // Each band names its own storage format; the arena its gid
                    // points into is a run of untyped bytes and cannot
                    // (`docs/archived/arena_unification.md` principle 8).
                    let bands: Vec<_> = cw.bands().collect();
                    for h in 0..n_kv_head {
                        let mut k_pal_slices = Vec::with_capacity(np);
                        let mut v_pal_slices = Vec::with_capacity(np);
                        for p in 0..np {
                            let base = (h * np + p) * 2;
                            let (k_gid, k_tag) = bands[base];
                            let (v_gid, v_tag) = bands[base + 1];

                            let k_data = read_band_chunk(
                                arenas,
                                k_gid,
                                k_tag,
                                chunk_size,
                                sub_head_dim,
                                device,
                            )?;
                            // Single-latent bands mix storage dtypes (FP8 nope ‖
                            // BF16 rope) — widen to F32 so the cats below join.
                            let k_data = if single_latent {
                                k_data.to_dtype(candle::DType::F32)?
                            } else {
                                k_data
                            };
                            // `read_band_chunk` returns (chunk_size, sub_head_dim),
                            // so the token is dim 0. The two unsqueezes rebuild
                            // the (1, head, token, sub_dim) shape the cats below
                            // join on.
                            k_pal_slices
                                .push(k_data.narrow(0, in_blk, 1)?.unsqueeze(0)?.unsqueeze(0)?);

                            let v_data = read_band_chunk(
                                arenas,
                                v_gid,
                                v_tag,
                                chunk_size,
                                sub_head_dim,
                                device,
                            )?;
                            let v_data = if single_latent {
                                v_data.to_dtype(candle::DType::F32)?
                            } else {
                                v_data
                            };
                            v_pal_slices
                                .push(v_data.narrow(0, in_blk, 1)?.unsqueeze(0)?.unsqueeze(0)?);
                        }
                        k_head_slices.push(LiveTensor::cat(&k_pal_slices, 3)?);
                        v_head_slices.push(LiveTensor::cat(&v_pal_slices, 3)?);
                    }
                    k_slices.push(LiveTensor::cat(&k_head_slices, 1)?);
                    v_slices.push(LiveTensor::cat(&v_head_slices, 1)?);
                }
            }

            // Concatenate along sequence dimension (dim 2), then take the
            // result off the arena.
            //
            // Every tensor above is a *lease* over arena bytes, and the arena
            // read-lock ends with this closure — after which another thread may
            // evict the slot and hand its bytes to a new tenant. `cat` copies
            // whenever it joins two or more pieces, but it short-circuits to
            // `arg0.clone()` for a single one, so on a one-chunk single-head
            // read the value that escapes here would be a lease over freed
            // storage. `to_owned_tensor` makes the copy unconditional, which is
            // what the `'static` in this function's return type has always
            // claimed.
            let k_out = LiveTensor::cat(&k_slices, 2)?.to_owned_tensor()?;
            let v_out = LiveTensor::cat(&v_slices, 2)?.to_owned_tensor()?;

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
        k: &candle::LiveTensor<'_>,
        v: &candle::LiveTensor<'_>,
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
        k: &candle::LiveTensor<'_>,
        v: &candle::LiveTensor<'_>,
    ) -> Result<()> {
        let (_, _, len, _) = k.dims4()?;

        let single_latent = self
            .inner
            .single_latent
            .load(std::sync::atomic::Ordering::Relaxed);

        // For quantized configs, arenas are created as F16 float (matching dequantize_f16).
        // For float configs, use the configured dtype.
        //
        // The single latent stores its bands across two arenas of DIFFERENT
        // dtypes (nope FP8 ‖ rope BF16), so a single up-front cast can't serve
        // both — and casting the whole latent to the nope FP8 dtype would strip
        // the rope tail's BF16 precision before it ever reaches its arena. Keep
        // the source precision here and cast each band to its own arena's dtype
        // at the slice_set below.
        let storage_dtype = self.inner.storage.dtype().unwrap_or(candle::DType::F16);

        let k = if single_latent || k.dtype() == storage_dtype {
            k.clone()
        } else {
            k.to_dtype(storage_dtype)?
        };
        let v = if single_latent || v.dtype() == storage_dtype {
            v.clone()
        } else {
            v.to_dtype(storage_dtype)?
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
            // COW any shared blocks in the write range. Whether a copy happened is
            // not interesting here: kernel block tables are built per launch from
            // the slot's current chunk list, so there is nothing to re-sync.
            self.ensure_blocks_writable_locked(&mut state, batch_idx, start_block, end_block)?;

            // **One launch for the whole write, when the bands allow it.**
            //
            // The loop below walks 32-token block × KV head × palette band, and
            // each step is a `narrow` + `contiguous` + a slot write: about 2,700
            // tiny GPU ops for a 649-token prefill on the 9B, which measured as
            // the single largest span in that prefill — more than the attention
            // it feeds, and flat in token count, i.e. launch-bound. The paged
            // prefill *kernel* never pays it because it scatters K/V itself;
            // only the float fallback writes from Rust.
            //
            // Every one of those copies is the same shape — `head_dim/N_PALETTE`
            // contiguous elements per token, `head_dim` apart at the source and
            // packed at the destination — so the whole write is one strided
            // migration plan. `try_plan_batched_write` returns `None` when a
            // band needs real work rather than a copy (a quantized tag, a
            // per-band dtype cast, a CPU arena), and then the walk below runs.
            #[cfg(feature = "cuda")]
            if let Some(plan) = self.try_plan_batched_write(
                &state,
                arena_state,
                batch_idx,
                offset,
                len,
                &k,
                &v,
                single_latent,
            )? {
                return super::migrate::kv_migrate_on(&self.inner.device, &plan, None);
            }

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
                // The single latent aliases V to K (the V bands are never stored,
                // see the `single_latent` skip below), so slicing/copying `v` here
                // is pure dead work — build the V segment only when it is written.
                let v_seg = if single_latent {
                    None
                } else {
                    Some(v.narrow(2, src_pos, seg)?.contiguous()?)
                };

                // Band path: each head has `n_palette` K/V sub-chunks of width
                // head_dim / n_palette (4 for GQA, LATENT_N_BANDS for the
                // single latent). Quantized K arenas (e.g. active R16) are
                // written via QTensor::quantize_into at the proper flat element offset.
                {
                    let np = self.inner.n_palette();
                    let sub_head_dim = (self.inner.head_dim / np).max(1);
                    // Snapshot the bands (gid + tag) before taking the mutable
                    // arena borrow the writes need.
                    let bands: Vec<(super::gid_pool::ChunkGid, ArenaFormatTag)> =
                        cw.bands().map(|(g, tag)| (g.clone(), tag)).collect();
                    let arenas = arena_state.arenas_mut();
                    for h in 0..n_kv_head {
                        let k_head = k_seg.narrow(1, h, 1)?.squeeze(1)?; // (1, seg, head_dim)
                        let v_head = match &v_seg {
                            Some(vs) => Some(vs.narrow(1, h, 1)?.squeeze(1)?), // (1, seg, head_dim)
                            None => None,
                        };

                        for p in 0..np {
                            let d_start = p * sub_head_dim;
                            let k_band = k_head.narrow(2, d_start, sub_head_dim)?.contiguous()?;

                            let base = (h * np + p) * 2;
                            let (k_gid, k_tag) = &bands[base];
                            let elem_offset = in_blk * sub_head_dim;
                            let elems = CHUNK_SIZE * sub_head_dim;

                            // Cast this band to its tag's storage dtype: the
                            // single latent's nope bands are FP8 and its rope
                            // bands BF16, so the band width alone doesn't fix
                            // the element type. A no-op for uniform-format
                            // (GQA) backings; quantized tags quantize from the
                            // float band inside `write_band_chunk`.
                            let k_band = match k_tag.to_kv_format() {
                                Some(KvFormat::Float(dt)) if k_band.dtype() != dt => {
                                    k_band.to_dtype(dt)?
                                }
                                _ => k_band,
                            };
                            write_band_chunk(arenas, k_gid, *k_tag, elems, elem_offset, &k_band)?;

                            if single_latent {
                                // K≡V: the V band aliases the K band just
                                // written — nothing separate to store.
                                continue;
                            }

                            // Reached only for real K/V backings (GQA); build the
                            // V band here so the single-latent path never pays it.
                            let v_band = v_head
                                .as_ref()
                                .expect("v_head present for a real K/V backing")
                                .narrow(2, d_start, sub_head_dim)?
                                .contiguous()?;
                            let (v_gid, v_tag) = &bands[base + 1];
                            let v_band = match v_tag.to_kv_format() {
                                Some(KvFormat::Float(dt)) if v_band.dtype() != dt => {
                                    v_band.to_dtype(dt)?
                                }
                                _ => v_band,
                            };
                            write_band_chunk(arenas, v_gid, *v_tag, elems, elem_offset, &v_band)?;
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
        // A slot holds ONE palette band — `chunk_size * (head_dim / N_PALETTE)`
        // elements — so a head's bytes are the concatenation of its four bands.
        //
        // This function used to size its read as `chunk_size * head_dim` and
        // issue it against palette 0's slot alone, reaching the other three
        // only because `alloc_chunk_run_for_key` *usually* lays a head's
        // palettes out contiguously. It does not always: a mixed-format band
        // group falls back to per-band allocation, and the read then walked
        // into unrelated chunks. Reading each band from its own gid removes the
        // dependency on the run layout — the same correction `resolve_band_source`
        // already made kernel-side. The slot bounds check is what surfaced it.
        let sub_head_dim = head_dim / N_PALETTE;
        let elems_per_band = chunk_size * sub_head_dim;
        // The bands' own tags say how many bytes each slot holds and whether it
        // is quantized at all. Palette 0 stands for the head, matching the
        // `k_gid(h)` / `v_gid(h)` aliases this function has always used.
        let bands: Vec<_> = cw.bands().map(|(g, tag)| (g.clone(), tag)).collect();

        self.inner.storage.read(|arena_state| {
            let arenas = arena_state.arenas();
            let mut k_bytes = Vec::new();
            let mut v_bytes = Vec::new();

            let read_band = |gid: &ChunkGid,
                             tag: ArenaFormatTag,
                             side: &str,
                             h: usize,
                             out: &mut Vec<u8>|
             -> Result<()> {
                let Some(KvFormat::Quantized(qf)) = tag.to_kv_format() else {
                    candle::bail!(
                        "read_raw_sealed_chunk: head {h} {side} is recorded as {tag:?}, not a \
                         quantized format (not yet reconciled); call reconcile before evicting"
                    )
                };
                let ggml = qf.to_ggml_dtype();
                let bytes = (elems_per_band / ggml.block_size()) * ggml.type_size();
                let arena = arenas.get(&gid.arena_idx()).ok_or_else(|| {
                    candle::Error::Msg(format!("arena {} not found", gid.arena_idx()))
                })?;
                out.extend_from_slice(&arena.slot_bytes(gid.chunk_idx(), bytes)?.to_vec1::<u8>()?);
                Ok(())
            };

            // Palette-major within each head — the order the four contiguous
            // bands used to produce, so the byte stream is unchanged.
            for h in 0..n_kv_head {
                for p in 0..N_PALETTE {
                    let base = h * GIDS_PER_HEAD + p * 2;
                    let (k_gid, k_tag) = &bands[base];
                    let (v_gid, v_tag) = &bands[base + 1];
                    read_band(k_gid, *k_tag, "K", h, &mut k_bytes)?;
                    read_band(v_gid, *v_tag, "V", h, &mut v_bytes)?;
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
        let np = self.inner.n_palette();
        let sub_head_dim = head_dim / np;

        // Snapshot each (head, palette) band — gid plus the tag that says how to
        // read it — while holding the state lock.
        type Band = (usize, usize, ArenaFormatTag);
        let head_gids: Vec<(Band, Band)> = {
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
            let flat: Vec<Band> = cw
                .bands()
                .map(|(g, tag)| (g.arena_idx(), g.chunk_idx(), tag))
                .collect();
            flat.chunks_exact(2).map(|kv| (kv[0], kv[1])).collect()
        };

        // Read each band under the storage lock, in `[t][pd]` (token-major)
        // order — what the interleave below expects.
        let bands: Vec<(Vec<f32>, Vec<f32>)> = self.inner.storage.read(|s| {
            let arenas = s.arenas();
            let mut bands = Vec::with_capacity(head_gids.len());
            for &((k_ai, k_ci, k_tag), (v_ai, v_ci, v_tag)) in head_gids.iter() {
                let read = |ai: usize, ci: usize, tag: ArenaFormatTag| -> Result<Vec<f32>> {
                    let arena = arenas
                        .get(&ai)
                        .ok_or_else(|| candle::Error::Msg(format!("arena {ai} not found")))?;
                    // Quantized bands (including R16) store each chunk
                    // DIM-major — block `pd` holds 32 tokens of dim `pd` — so
                    // the dequantized flat is `[pd][t]` and has to be
                    // transposed. Float bands are already `[t][pd]`.
                    let elems = chunk_size * sub_head_dim;
                    match tag.to_kv_format() {
                        Some(KvFormat::Quantized(qf)) => {
                            let ggml = qf.to_ggml_dtype();
                            let bytes = arena
                                .slot_bytes(ci, (elems / ggml.block_size()) * ggml.type_size())?
                                .to_vec1::<u8>()?;
                            qtensor_from_ggml(ggml, &bytes, vec![elems], &Device::Cpu)?
                                .dequantize(&Device::Cpu)?
                                .reshape((sub_head_dim, chunk_size))?
                                .transpose(0, 1)?
                                .contiguous()?
                                .flatten_all()?
                                .to_vec1::<f32>()
                        }
                        Some(KvFormat::Float(dtype)) => arena
                            .read_slot_typed(ci, dtype, (chunk_size, sub_head_dim))?
                            .to_dtype(candle::DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>(),
                        None => candle::bail!(
                            "read_f32_sampler_chunk: band tag {tag:?} names no storage format"
                        ),
                    }
                };
                bands.push((read(k_ai, k_ci, k_tag)?, read(v_ai, v_ci, v_tag)?));
            }
            Ok::<_, candle::Error>(bands)
        })??;

        // Transpose each sub-band from [t][pd] → [pd][t] and interleave into
        // the full head_dim layout: out[(h * head_dim + d) * chunk_size + t].
        let mut k_out = vec![0.0f32; n_kv_head * head_dim * chunk_size];
        let mut v_out = vec![0.0f32; n_kv_head * head_dim * chunk_size];
        for (idx, (k_flat, v_flat)) in bands.into_iter().enumerate() {
            let h = idx / np;
            let p = idx % np;
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
