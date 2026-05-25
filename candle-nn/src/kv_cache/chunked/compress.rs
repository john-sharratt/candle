//! Decode-path quantization compression for KV cache chunks.
//!
//! This module contains the v2 reconcile path that is invoked during decode
//! (as opposed to prefill). The v2 path now derives the real per-palette K/V
//! layout from sampled selection and persists that routing through the palette
//! maps consumed by the decode and prefill kernels.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle::quantized::cuda::{dtype_to_ggml_float, identity_pal_map_128, PalHeadDesc};
#[cfg(feature = "cuda")]
use candle::quantized::GgmlDType;
#[cfg(feature = "cuda")]
use candle::Device;
#[cfg(feature = "cuda")]
use candle::Result;

#[cfg(feature = "cuda")]
use super::head_gids::{HeadGids, GIDS_PER_HEAD};
#[cfg(feature = "cuda")]
use super::sampled_selection::{PagedSelectionGpuInputs, SampleFormat};
#[cfg(feature = "cuda")]
use super::{
    ArenaKey, ChunkedKvBacking, CompressionPolicy, CHUNK_SIZE, PRODUCTION_K_QREL_HIGH_THRESHOLDS,
    PRODUCTION_K_QREL_LOW_THRESHOLDS, PRODUCTION_V_QREL_HIGH_THRESHOLDS,
    PRODUCTION_V_QREL_LOW_THRESHOLDS,
};
#[cfg(feature = "cuda")]
use crate::kv_cache::arena_table::N_PALETTE;
#[cfg(feature = "cuda")]
use crate::kv_cache::chunked::backing::BackingInner;
#[cfg(feature = "cuda")]
use crate::kv_cache::chunked::bg_quantizer::ChunkMigration;
#[cfg(feature = "cuda")]
use crate::kv_cache::{KvFormat, QuantFormat};
#[cfg(feature = "cuda")]
use crate::ChunkGid;
#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::CudaStream;
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::Generation;

#[cfg(feature = "cuda")]
impl ChunkedKvBacking {
    fn pack_head_palette_maps(
        assignments: &[u8],
        n_batch: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let packed_bytes = head_dim / 4;
        let expected_packed_bytes = identity_pal_map_128().len();
        if head_dim % 4 != 0 || head_dim % N_PALETTE != 0 || packed_bytes != expected_packed_bytes {
            candle::bail!(
                "reconcile_batch_float_to_quant_v2: palette4 conversion requires head_dim={}, got {}",
                expected_packed_bytes * 4,
                head_dim
            );
        }

        let dims_per_head = head_dim;
        let expected = n_batch
            .checked_mul(n_head)
            .and_then(|v| v.checked_mul(dims_per_head))
            .ok_or_else(|| candle::Error::Msg("palette map shape overflow".into()))?;
        if assignments.len() != expected {
            candle::bail!(
                "palette map length mismatch: got {}, expected {} ({} dims/head × {} batch-heads)",
                assignments.len(),
                expected,
                dims_per_head,
                n_batch * n_head
            );
        }

        let target_per_palette = head_dim / N_PALETTE;
        let mut head_palette_maps = Vec::with_capacity(n_batch);
        for block_i in 0..n_batch {
            let mut block_map = vec![0u8; n_head * packed_bytes];
            for h in 0..n_head {
                let base = (block_i * n_head + h) * dims_per_head;
                let mut counts = [0usize; N_PALETTE];
                for (dim_offset, &raw_slot) in
                    assignments[base..base + dims_per_head].iter().enumerate()
                {
                    let slot = (raw_slot & 0x3) as usize;
                    counts[slot] += 1;
                    let byte_idx = h * packed_bytes + (dim_offset / 4);
                    block_map[byte_idx] |= (slot as u8) << (2 * (dim_offset % 4));
                }
                if counts.iter().any(|&count| count != target_per_palette) {
                    candle::bail!(
                        "invalid palette4 routing for block {} head {}: {:?}",
                        block_i,
                        h,
                        counts
                    );
                }
            }
            head_palette_maps.push(block_map);
        }
        Ok(head_palette_maps)
    }
}

#[cfg(feature = "cuda")]
impl BackingInner {
    /// Batched GPU Float→Quant migration path (v2).
    ///
    /// Adaptive format selection now lives inside this function and is driven by
    /// the existing sampled-selection workflow rather than a duplicated manual
    /// selection path in the callers.
    pub(super) fn reconcile_batch_float_to_quant_v2(
        &self,
        arc_self: &Arc<BackingInner>,
        migrations: Vec<ChunkMigration>,
        compression: Option<&CompressionPolicy>,
        generation: &Generation,
        stream: &Arc<CudaStream>,
        sync_before_gid_update: bool,
    ) -> Result<usize> {
        if migrations.is_empty() {
            return Ok(0);
        }

        // The palette4 CUDA kernel is compiled for KVHEAD_HD=128 only.
        // Models with a different head_dim (e.g. Qwen2-0.5B with head_dim=64)
        // cannot use KV compression — callers must skip these models.
        {
            let expected_head_dim = identity_pal_map_128().len() * 4;
            if self.head_dim != expected_head_dim {
                candle::bail!(
                    "KV cache compression requires head_dim={}, but this model has head_dim={}. \
                     Use --compression none for models with head_dim != 128.",
                    expected_head_dim,
                    self.head_dim
                );
            }
        }

        let target_key = self.storage.default_key();
        let v_target_key = Some(self.storage.default_v_key());

        let mut valid: Vec<ChunkMigration> = Vec::new();
        {
            let layer_tree = self.layer_tree();
            let layer_tree = layer_tree
                .iter()
                .map(|(a, b)| (a.clone(), b.read().unwrap()))
                .collect::<ahash::HashMap<_, _>>();

            let _ = self.storage.read(|storage| {
                for m in migrations {
                    let layer_idx = m.layer_idx;
                    let state = layer_tree.get(&layer_idx).ok_or_else(|| {
                        candle::Error::Msg(format!("layer {layer_idx} not registered"))
                    })?;
                    let block_count = state
                        .sequences
                        .get(m.batch_idx)
                        .and_then(|s| s.as_ref())
                        .map(|slot| slot.block_count())
                        .unwrap_or(0);
                    if block_count == 0 || m.block_idx + 1 > block_count {
                        continue;
                    }

                    let first_k_src = m.head_gids.k_gid(0);
                    let first_k_ai = first_k_src.arena_idx();
                    if !matches!(
                        storage
                            .arenas()
                            .get(&first_k_ai)
                            .map(|a| a.format())
                            .unwrap_or(KvFormat::Float(candle::DType::F16)),
                        KvFormat::Float(_) | KvFormat::Quantized(QuantFormat::R16)
                    ) {
                        continue;
                    }

                    valid.push(m);
                }
                Ok::<(), candle::Error>(())
            })?;
        }

        if valid.is_empty() {
            return Ok(0);
        }

        let n_batch = valid.len();
        let n_head = self.n_kv_head;
        let head_dim = self.head_dim;

        let k_head_palette_formats: Vec<Vec<QuantFormat>>;
        let v_head_palette_formats: Vec<Vec<QuantFormat>>;
        let k_head_palette_scale: Vec<Vec<f32>>;
        let v_head_palette_scale: Vec<Vec<f32>>;
        let k_head_palette_maps: Vec<Vec<u8>>;
        let v_head_palette_maps: Vec<Vec<u8>>;

        // Adaptive selection runs whenever a compression policy is supplied.
        // Without a policy (`None`), we fill in default routing (uniform
        // target format, unit scale, identity palette map) right here so the
        // convert function consumes pre-filled values uniformly. Per-block
        // scales always reach the chunk — when no selection ran, they're 1.0.
        if let Some(compression) = compression {
            let (k_profile, v_profile) =
                CompressionPolicy::production_candidates(compression.compression_level);
            let k_candidates: Vec<SampleFormat> = k_profile
                .into_iter()
                .filter_map(SampleFormat::from_kv_format)
                .filter(|fmt| !fmt.is_float())
                .collect();
            let v_candidates: Vec<SampleFormat> = v_profile
                .into_iter()
                .filter_map(SampleFormat::from_kv_format)
                .filter(|fmt| !fmt.is_float())
                .collect();

            if k_candidates.is_empty() || v_candidates.is_empty() {
                candle::bail!(
                    "adaptive v2 quantization is missing shared production candidates for level {}",
                    compression.compression_level
                );
            }

            {
                let cuda_dev = match &self.device {
                    Device::Cuda(d) => d,
                    _ => candle::bail!("reconcile_batch_float_to_quant_v2: requires CUDA device"),
                };
                let chunk_gids: Vec<HeadGids> =
                    valid.iter().cloned().map(|m| m.head_gids).collect();
                let gpu_inputs = PagedSelectionGpuInputs::from_head_gids(
                    arc_self.clone(),
                    &chunk_gids,
                    Some(generation),
                    cuda_dev,
                )?;

                let expected_head_dim = identity_pal_map_128().len() * 4;
                if head_dim != expected_head_dim {
                    candle::bail!(
                        "adaptive palette4 conversion requires head_dim={}, got {}",
                        expected_head_dim,
                        head_dim
                    );
                }
                let threshold_idx = compression.compression_level.min(10) as usize;
                let to_quant = |fmt: SampleFormat| {
                    fmt.to_quant_format().ok_or_else(|| {
                        candle::Error::Msg(
                            format!(
                                "adaptive v2 quantization produced non-quant palette format {fmt}"
                            )
                            .into(),
                        )
                    })
                };
                let reject_palette_float_formats = |stage: &str,
                                                    rows: &[[SampleFormat; 4]]|
                 -> Result<()> {
                    for row in rows {
                        if let Some(fmt) = row.iter().copied().find(|fmt| fmt.is_float()) {
                            candle::bail!(
                                    "adaptive v2 quantization produced non-quant palette format {} during {}",
                                    fmt,
                                    stage
                                );
                        }
                    }
                    Ok(())
                };

                let k_threshold_hi = PRODUCTION_K_QREL_HIGH_THRESHOLDS[threshold_idx]
                    * compression.k_hi_error_threshold_factor;
                let k_threshold_lo = PRODUCTION_K_QREL_LOW_THRESHOLDS[threshold_idx]
                    * compression.k_low_error_threshold_factor;
                let v_threshold_hi = PRODUCTION_V_QREL_HIGH_THRESHOLDS[threshold_idx]
                    * compression.v_hi_error_threshold_factor;
                let v_threshold_lo = PRODUCTION_V_QREL_LOW_THRESHOLDS[threshold_idx]
                    * compression.v_low_error_threshold_factor;

                let (
                    k_palette4_rows,
                    v_palette4_rows,
                    k_palette_scale_rows,
                    v_palette_scale_rows,
                    k_assignments,
                    v_assignments,
                    _k_head_amax_gpu,
                    _v_head_amax_gpu,
                ) = gpu_inputs.select_palette4_formats_fused(
                    &k_candidates,
                    &v_candidates,
                    k_threshold_hi,
                    k_threshold_lo,
                    v_threshold_hi,
                    v_threshold_lo,
                    Some(generation),
                )?;
                reject_palette_float_formats("fused selection (K)", &k_palette4_rows)?;
                reject_palette_float_formats("fused selection (V)", &v_palette4_rows)?;

                k_head_palette_formats = (0..n_batch)
                    .map(|chunk_idx| {
                        let start = chunk_idx * n_head;
                        let end = start + n_head;
                        k_palette4_rows[start..end]
                            .iter()
                            .flat_map(|row| row.iter().copied())
                            .map(to_quant)
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?;
                v_head_palette_formats = (0..n_batch)
                    .map(|chunk_idx| {
                        let start = chunk_idx * n_head;
                        let end = start + n_head;
                        v_palette4_rows[start..end]
                            .iter()
                            .flat_map(|row| row.iter().copied())
                            .map(to_quant)
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?;
                k_head_palette_scale = (0..n_batch)
                    .map(|chunk_idx| {
                        let start = chunk_idx * n_head;
                        let end = start + n_head;
                        k_palette_scale_rows[start..end]
                            .iter()
                            .flat_map(|row| row.iter().copied())
                            .collect::<Vec<f32>>()
                    })
                    .collect::<Vec<_>>();
                v_head_palette_scale = (0..n_batch)
                    .map(|chunk_idx| {
                        let start = chunk_idx * n_head;
                        let end = start + n_head;
                        v_palette_scale_rows[start..end]
                            .iter()
                            .flat_map(|row| row.iter().copied())
                            .collect::<Vec<f32>>()
                    })
                    .collect::<Vec<_>>();
                k_head_palette_maps = ChunkedKvBacking::pack_head_palette_maps(
                    &k_assignments,
                    n_batch,
                    n_head,
                    head_dim,
                )?;
                v_head_palette_maps = ChunkedKvBacking::pack_head_palette_maps(
                    &v_assignments,
                    n_batch,
                    n_head,
                    head_dim,
                )?;
            }
        } else {
            // No compression policy — fill defaults: target format per palette,
            // unit (1.0) outer scale, identity palette map. The convert function
            // consumes these directly without any fallback logic.
            let target_qfmt = match target_key.format {
                KvFormat::Quantized(qf) => qf,
                KvFormat::Float(_) => candle::bail!(
                    "reconcile_batch_float_to_quant_v2 requires a quantized target_key when no \
                     compression policy is supplied (got {:?})",
                    target_key.format
                ),
            };
            let v_target_qfmt = match v_target_key.as_ref().map(|k| k.format.clone()) {
                Some(KvFormat::Quantized(qf)) => qf,
                Some(KvFormat::Float(f)) => candle::bail!(
                    "reconcile_batch_float_to_quant_v2: v_target must be quantized without a \
                     compression policy (got Float({:?}))",
                    f
                ),
                None => target_qfmt,
            };
            let k_unit_fmts: Vec<QuantFormat> = vec![target_qfmt; n_head * N_PALETTE];
            let v_unit_fmts: Vec<QuantFormat> = vec![v_target_qfmt; n_head * N_PALETTE];
            let unit_scale: Vec<f32> = vec![1.0f32; n_head * N_PALETTE];
            let identity_pal_row: Vec<u8> = (*self.identity_pal).clone();

            k_head_palette_formats = vec![k_unit_fmts; n_batch];
            v_head_palette_formats = vec![v_unit_fmts; n_batch];
            k_head_palette_scale = vec![unit_scale.clone(); n_batch];
            v_head_palette_scale = vec![unit_scale; n_batch];
            k_head_palette_maps = vec![identity_pal_row.clone(); n_batch];
            v_head_palette_maps = vec![identity_pal_row; n_batch];
        }

        self.convert_valid_float_to_quant_palette4(
            valid,
            &k_head_palette_formats,
            &v_head_palette_formats,
            &k_head_palette_scale,
            &v_head_palette_scale,
            &k_head_palette_maps,
            &v_head_palette_maps,
            generation,
            stream,
            sync_before_gid_update,
        )
    }

    /// Run the palette4 float→quant convert kernel with pre-filled per-(block,
    /// head, palette) routing. The caller must populate every input array to
    /// length `valid.len()` × `n_kv_head` × `N_PALETTE` (or `n_kv_head ×
    /// pal_bytes` for the palette maps); this function trusts those shapes.
    pub(super) fn convert_valid_float_to_quant_palette4(
        &self,
        migrations: Vec<ChunkMigration>,
        k_head_palette_formats: &[Vec<QuantFormat>],
        v_head_palette_formats: &[Vec<QuantFormat>],
        k_head_palette_scale: &[Vec<f32>],
        v_head_palette_scale: &[Vec<f32>],
        k_head_palette_maps: &[Vec<u8>],
        v_head_palette_maps: &[Vec<u8>],
        generation: &Generation,
        stream: &std::sync::Arc<CudaStream>,
        sync_before_gid_update: bool,
    ) -> Result<usize> {
        if migrations.is_empty() {
            return Ok(0);
        }

        let n_kv_head = self.n_kv_head;
        let head_dim = self.head_dim;
        let sub_head_dim = head_dim / N_PALETTE;
        let elems_per_head = CHUNK_SIZE * sub_head_dim;

        let k_format_for = |block_i: usize, h: usize, p: usize| -> QuantFormat {
            k_head_palette_formats[block_i][h * N_PALETTE + p]
        };
        let v_format_for = |block_i: usize, h: usize, p: usize| -> QuantFormat {
            v_head_palette_formats[block_i][h * N_PALETTE + p]
        };

        let fmt_to_key = |qf: QuantFormat| ArenaKey::gpu_quant(qf);

        // The caller has already filtered this down to eligible float/R16
        // source blocks. What remains here is the actual palette4 conversion
        // and the persistence of the selected per-palette routing metadata.
        let valid = migrations;

        // Phase 2: Allocate destination chunks for exactly the valid migrations.
        #[derive(Clone)]
        struct Pending {
            batch_idx: usize,
            layer_idx: usize,
            block_idx: usize,
            old_gids: HeadGids,
            /// Flat layout [h][p][k, v]: len = n_kv_head * GIDS_PER_HEAD
            new_gids: Vec<ChunkGid>,
        }

        let mut pending: Vec<Pending> = Vec::with_capacity(valid.len());
        {
            for (block_i, m) in valid.iter().enumerate() {
                let mut new_gids = Vec::with_capacity(n_kv_head * GIDS_PER_HEAD);
                for h in 0..n_kv_head {
                    for p in 0..N_PALETTE {
                        let k_gid =
                            self.alloc_chunk_for_key(fmt_to_key(k_format_for(block_i, h, p)))?;
                        let v_gid =
                            self.alloc_chunk_for_key(fmt_to_key(v_format_for(block_i, h, p)))?;
                        new_gids.push(k_gid);
                        new_gids.push(v_gid);
                    }
                }
                pending.push(Pending {
                    batch_idx: m.batch_idx,
                    layer_idx: m.layer_idx,
                    block_idx: m.block_idx,
                    old_gids: m.head_gids.clone(),
                    new_gids,
                });
            }
        }

        // Phase 3: Build PalHeadDesc structs (one per migration × head) and GPU pointers.
        // Re-check tombstone here as a safety guard against races; any GIDs allocated
        // for a skipped migration are recycled automatically via ChunkGid's Drop impl.
        let mut descs: Vec<PalHeadDesc> = Vec::new();
        let mut completed: Vec<(
            usize,
            Pending,
            std::sync::Arc<Vec<f32>>,
            std::sync::Arc<Vec<f32>>,
        )> = Vec::new();

        // R16 byte stride per palette-head: (PAL_DIM * num_chunks) blocks × 128 bytes/block
        let r16_bytes_per_head = (elems_per_head / 32) * 128;

        self.storage.try_write(|storage| {
            for (block_i, m) in pending.iter().enumerate() {
                // Safety re-check: tombstone status can change between phases.
                let ident = identity_pal_map_128();

                for h in 0..n_kv_head {
                    let mut k_src_ptrs = [0u64; N_PALETTE];
                    let mut k_dst_ptrs = [0u64; N_PALETTE];
                    let mut k_src_fmts = [GgmlDType::F16; N_PALETTE];
                    let mut k_dst_fmts = [GgmlDType::F16; N_PALETTE];
                    let mut v_src_ptrs = [0u64; N_PALETTE];
                    let mut v_dst_ptrs = [0u64; N_PALETTE];
                    let mut v_src_fmts = [GgmlDType::F16; N_PALETTE];
                    let mut v_dst_fmts = [GgmlDType::F16; N_PALETTE];

                    for p in 0..N_PALETTE {
                        let gid_slot = h * GIDS_PER_HEAD + p * 2;

                        // K source (float or R16)
                        let k_src_gid = m.old_gids.k_gid_pal(h, p);
                        let k_src_ai = k_src_gid.arena_idx();
                        let k_src_arena = storage.arenas().get(&k_src_ai).ok_or_else(|| {
                            candle::Error::Msg(format!("k src arena {k_src_ai} not found"))
                        })?;
                        let k_src_is_r16 =
                            matches!(k_src_arena.format(), KvFormat::Quantized(QuantFormat::R16));
                        k_src_fmts[p] = if k_src_is_r16 {
                            GgmlDType::R16
                        } else {
                            dtype_to_ggml_float(k_src_arena.float_data()?.dtype())?
                        };
                        k_src_ptrs[p] = if k_src_is_r16 {
                            let qt = k_src_arena.quantized_data()?;
                            ChunkedKvBacking::qtensor_ptr_at_byte_offset(
                                qt,
                                k_src_gid.chunk_idx() * r16_bytes_per_head,
                            )?
                        } else {
                            ChunkedKvBacking::tensor_ptr_at_offset(
                                k_src_arena.float_data()?,
                                k_src_gid.chunk_idx() * elems_per_head,
                            )?
                        };

                        // K destination for this migrated chunk's head/palette layout
                        let k_dst_gid = &m.new_gids[gid_slot];
                        {
                            let qf = k_format_for(block_i, h, p);
                            let qdtype = qf.to_ggml_dtype();
                            let bytes_per_chunk =
                                (elems_per_head / qdtype.block_size()) * qdtype.type_size();
                            let k_dst_ai = k_dst_gid.arena_idx();
                            let k_dst_data = storage
                                .arenas()
                                .get(&k_dst_ai)
                                .ok_or_else(|| {
                                    candle::Error::Msg(format!("k dst arena {k_dst_ai} not found"))
                                })?
                                .quantized_data()?;
                            k_dst_fmts[p] = qdtype;
                            k_dst_ptrs[p] = ChunkedKvBacking::qtensor_ptr_at_byte_offset(
                                k_dst_data,
                                k_dst_gid.chunk_idx() * bytes_per_chunk,
                            )?;
                        }

                        // V source (float or R16)
                        let v_src_gid = m.old_gids.v_gid_pal(h, p);
                        let v_src_ai = v_src_gid.arena_idx();
                        let v_src_arena = storage.arenas().get(&v_src_ai).ok_or_else(|| {
                            candle::Error::Msg(format!("v src arena {v_src_ai} not found"))
                        })?;
                        let v_src_is_r16 =
                            matches!(v_src_arena.format(), KvFormat::Quantized(QuantFormat::R16));
                        v_src_fmts[p] = if v_src_is_r16 {
                            GgmlDType::R16
                        } else {
                            dtype_to_ggml_float(v_src_arena.float_data()?.dtype())?
                        };
                        v_src_ptrs[p] = if v_src_is_r16 {
                            let qt = v_src_arena.quantized_data()?;
                            ChunkedKvBacking::qtensor_ptr_at_byte_offset(
                                qt,
                                v_src_gid.chunk_idx() * r16_bytes_per_head,
                            )?
                        } else {
                            ChunkedKvBacking::tensor_ptr_at_offset(
                                v_src_arena.float_data()?,
                                v_src_gid.chunk_idx() * elems_per_head,
                            )?
                        };

                        // V destination for this migrated chunk's head/palette layout
                        let v_dst_gid = &m.new_gids[gid_slot + 1];
                        {
                            let qf = v_format_for(block_i, h, p);
                            let qdtype = qf.to_ggml_dtype();
                            let bytes_per_chunk =
                                (elems_per_head / qdtype.block_size()) * qdtype.type_size();
                            let v_dst_ai = v_dst_gid.arena_idx();
                            let v_dst_data = storage
                                .arenas()
                                .get(&v_dst_ai)
                                .ok_or_else(|| {
                                    candle::Error::Msg(format!("v dst arena {v_dst_ai} not found"))
                                })?
                                .quantized_data()?;
                            v_dst_fmts[p] = qdtype;
                            v_dst_ptrs[p] = ChunkedKvBacking::qtensor_ptr_at_byte_offset(
                                v_dst_data,
                                v_dst_gid.chunk_idx() * bytes_per_chunk,
                            )?;
                        }
                    }

                    let pal_bytes = ident.len();
                    let pal_start = h * pal_bytes;
                    let pal_end = pal_start + pal_bytes;
                    let mut k_dst_pal_map = ident;
                    k_dst_pal_map
                        .copy_from_slice(&k_head_palette_maps[block_i][pal_start..pal_end]);
                    let mut v_dst_pal_map = ident;
                    v_dst_pal_map
                        .copy_from_slice(&v_head_palette_maps[block_i][pal_start..pal_end]);

                    let mut k_dst_scales = [1.0f32; N_PALETTE];
                    let mut v_dst_scales = [1.0f32; N_PALETTE];
                    let scale_start = h * N_PALETTE;
                    k_dst_scales.copy_from_slice(
                        &k_head_palette_scale[block_i][scale_start..scale_start + N_PALETTE],
                    );
                    v_dst_scales.copy_from_slice(
                        &v_head_palette_scale[block_i][scale_start..scale_start + N_PALETTE],
                    );

                    descs.push(PalHeadDesc {
                        k_src_arena_ptrs: k_src_ptrs,
                        v_src_arena_ptrs: v_src_ptrs,
                        k_src_fmts,
                        v_src_fmts,
                        k_src_pal_map: ident,
                        v_src_pal_map: ident,
                        // Source is always float-typed here (this function
                        // converts valid_float → quant), so no outer scale is
                        // baked into the source arena.
                        k_src_scales: [1.0f32; N_PALETTE],
                        v_src_scales: [1.0f32; N_PALETTE],
                        k_dst_arena_ptrs: k_dst_ptrs,
                        v_dst_arena_ptrs: v_dst_ptrs,
                        k_dst_fmts,
                        v_dst_fmts,
                        k_dst_pal_map,
                        v_dst_pal_map,
                        k_dst_scales,
                        v_dst_scales,
                    });
                }

                completed.push((
                    block_i,
                    m.clone(),
                    std::sync::Arc::new(k_head_palette_scale[block_i].clone()),
                    std::sync::Arc::new(v_head_palette_scale[block_i].clone()),
                ));
            }
            Ok(())
        })?;

        if completed.is_empty() {
            return Ok(0);
        }

        // Phase 4 preamble: ensure stream sees dest arena allocations made on
        // the primary stream (Phase 2).  Without this barrier the convert kernel
        // can fault on addresses that primary_stream hasn't yet made valid.
        {
            let Device::Cuda(cuda_dev) = &self.device else {
                candle::bail!("convert_valid_float_to_quant_palette4: CUDA device required");
            };
            let primary_stream = cuda_dev.cuda_stream();
            let alloc_event = primary_stream
                .record_event(None)
                .map_err(|e| candle::Error::Msg(format!("alloc_event record: {e:?}")))?;
            stream
                .wait(&alloc_event)
                .map_err(|e| candle::Error::Msg(format!("stream wait alloc_event: {e:?}")))?;
        }

        // Phase 4: kernel launch — grid = (n_kv_head, num_migrations).
        candle::quantized::cuda::quantize_palette4_convert_buffered(
            &descs,
            n_kv_head,
            completed.len(),
            1,
            generation,
            stream,
        )?;

        // Sync stream before updating GIDs so the decode kernel never reads an
        // arena slot before quant data lands (needed when stream != primary).
        if sync_before_gid_update {
            stream
                .synchronize()
                .map_err(|e| candle::Error::Msg(format!("stream sync failed: {e:?}")))?;
        }

        // Phase 5: Update block table + sync GPU arena pointer buffer.
        // Persist the exact palette routing used by the conversion kernel so
        // decode/prefill read back with the same head/palette layout.
        let ret = completed.len();
        let arena_info = self.resolve_arena_info()?;
        for (block_i, pending, k_scale_arc, v_scale_arc) in completed.into_iter() {
            let k_pal = std::sync::Arc::new(k_head_palette_maps[block_i].clone());
            let v_pal = std::sync::Arc::new(v_head_palette_maps[block_i].clone());
            let new_gids = HeadGids::from_vec(pending.new_gids);
            if pending.layer_idx == 0 {
                let k_ai0 = new_gids.k_gid(0).arena_idx();
                let v_ai0 = new_gids.v_gid(0).arena_idx();
                tracing::debug!(
                    "bg_quantizer: set_block_gids layer={l} batch={b} blk={blk} k_arena={k_ai0} v_arena={v_ai0}",
                    l = pending.layer_idx,
                    b = pending.batch_idx,
                    blk = pending.block_idx
                );
            }
            self.set_block_gids_sharded_and_update_gpu(
                pending.layer_idx,
                pending.batch_idx,
                pending.block_idx,
                new_gids,
                k_pal,
                v_pal,
                k_scale_arc,
                v_scale_arc,
                &arena_info,
            )?;
        }

        Ok(ret)
    }
}
