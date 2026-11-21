//! Quantizes one layer's sealed chunks to the policy's formats, with the format
//! selection and the convert launch batched across every layer.
//!
//! A layer's chunks split into the ones the kernels can read, GPU float or R16,
//! and a preserve bucket that passes through unchanged, such as a view of
//! cold-loaded chunks already in mixed quantized formats. The selection kernel
//! runs once over the flat list of every layer's chunks, cut into runs that fit
//! the persistence staging, and each layer then takes its own slice. A layer
//! refused mid-wave owes the demand of every layer after it, so the retry
//! claims the whole group at once. The deferred form appends each layer's
//! convert to a shared accumulator and gives output bit-identical to converting
//! layer by layer.

use super::*;

/// What a policy hands the format selection: its level's quantized K and V
/// candidates and the four error thresholds `[k_hi, k_lo, v_hi, v_lo]` — the
/// production table's row for the level scaled by the policy's factors. Fixed
/// by level and model, never per block.
#[cfg(feature = "cuda")]
pub(in super::super) struct PolicySelection {
    pub k: Vec<SampleFormat>,
    pub v: Vec<SampleFormat>,
    pub thresholds: [f32; 4],
}

#[cfg(feature = "cuda")]
pub(in super::super) fn policy_selection(policy: &CompressionPolicy) -> Result<PolicySelection> {
    let (k_profile, v_profile) =
        super::super::compression_policy::production_adaptive_candidates(policy.compression_level);
    let quantized = |profile: Vec<KvFormat>| -> Vec<SampleFormat> {
        profile
            .into_iter()
            .filter_map(SampleFormat::from_kv_format)
            .filter(|f| !f.is_float())
            .collect()
    };
    let (k, v) = (quantized(k_profile), quantized(v_profile));
    if k.is_empty() || v.is_empty() {
        candle::bail!(
            "missing K/V production candidates for level {}",
            policy.compression_level
        );
    }
    let ti = policy.compression_level.min(10) as usize;
    Ok(PolicySelection {
        k,
        v,
        thresholds: [
            PRODUCTION_K_QREL_HIGH_THRESHOLDS[ti] * policy.k_hi_error_threshold_factor,
            PRODUCTION_K_QREL_LOW_THRESHOLDS[ti] * policy.k_low_error_threshold_factor,
            PRODUCTION_V_QREL_HIGH_THRESHOLDS[ti] * policy.v_hi_error_threshold_factor,
            PRODUCTION_V_QREL_LOW_THRESHOLDS[ti] * policy.v_low_error_threshold_factor,
        ],
    })
}

/// Cross-layer batched form: quantize this layer but **do not** launch the
/// convert — append its per-(chunk, head) descriptors to `descs_acc` instead.
/// The caller runs ONE convert over every layer's accumulated descriptors via
/// [`convert_deferred_descs`] before the produced bytes are read, collapsing the
/// per-layer convert launches (the WDDM-bound cost that dominates the hot→warm
/// drain). The returned sequences already point at the allocated dst arenas; the
/// deferred convert fills them. Bit-identical to converting per layer.
#[cfg(feature = "cuda")]
pub fn quantize_sealed_in_place_deferred(
    backing: &ChunkedKvBacking,
    sequences: &[&SealedSequence],
    policy: &CompressionPolicy,
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    pinned_scratch: &mut Option<PinnedBuf>,
    descs_acc: &mut Vec<PalHeadDesc>,
) -> Result<Vec<SealedSequence>> {
    quantize_sealed_in_place_impl(
        backing,
        sequences,
        policy,
        device,
        copy_stream,
        pinned_scratch,
        Some(descs_acc),
        None,
        None,
    )
}

/// One layer's chunks partitioned for quantization: the kernel-eligible jobs
/// (full/partial GPU-`Float` or `R16` chunks) and the preserve bucket (mixed-Q
/// chunks passed through unchanged). Owned, so it outlives the storage read lock
/// and can be threaded through a cross-layer selection.
#[cfg(feature = "cuda")]
pub(super) struct QuantBucket {
    /// One entry per kernel-eligible chunk: its band gids **and** the band
    /// format tags that say how to read them. The selection table and the
    /// convert-descriptor build both need the pair; the arena a gid points into
    /// can only answer the first half.
    pub(super) chunk_jobs: Vec<ChunkBands>,
    pub(super) chunk_valid_ranges: Vec<i32>,
    pub(super) seq_chunk_map: Vec<(usize, usize)>,
    pub(super) preserve_per_seq: Vec<Vec<(usize, SealedChunk)>>,
}

/// Partition `sequences`' chunks into kernel-eligible jobs vs the preserve
/// bucket. A chunk is eligible when every `(h, p)` source band is GPU-resident
/// and recorded as `Float` or `Quantized(R16)` — the formats the selection +
/// convert kernels read directly. Location comes from the arena (it is arena
/// identity); the format comes from the chunk's own band tags, because under
/// size classes an arena holds whatever fits its stride and cannot answer the
/// question. Full and partial chunks both qualify: the kernels read a partial
/// chunk's dead slots as zero whatever they hold, and the packed valid range
/// corrects the count-normalized metrics. Ineligible chunks (e.g. a
/// view borrowing cold-loaded mixed-Q chunks) go to `preserve` and merge back
/// unchanged. `Ok(None)` ⇒ nothing eligible (caller passes the sequences
/// through verbatim).
#[cfg(feature = "cuda")]
pub(super) fn bucket_quant_chunks(
    backing: &ChunkedKvBacking,
    sequences: &[&SealedSequence],
) -> Result<Option<QuantBucket>> {
    let mut chunk_jobs: Vec<ChunkBands> = Vec::new();
    let mut chunk_valid_ranges: Vec<i32> = Vec::new();
    let mut seq_chunk_map: Vec<(usize, usize)> = Vec::new();
    let mut preserve_per_seq: Vec<Vec<(usize, SealedChunk)>> =
        sequences.iter().map(|_| Vec::new()).collect();
    backing.inner.storage.read(|storage| {
        for (seq_idx, seq) in sequences.iter().enumerate() {
            for (chunk_idx, chunk) in seq.chunks.iter().enumerate() {
                let mut all_gids_kernel_eligible = true;
                for (gid, tag) in chunk.bands() {
                    let ok = super::super::chunk_ops::needs_reconcile_source_tag(tag)
                        && matches!(
                            storage.arena_key(gid.arena_idx()),
                            Some(k) if k.location == ArenaLocation::Gpu
                        );
                    if !ok {
                        all_gids_kernel_eligible = false;
                        break;
                    }
                }
                if all_gids_kernel_eligible {
                    chunk_jobs.push(ChunkBands::from_sealed(chunk));
                    let len = usize::from(chunk.token_count).clamp(1, CHUNK_SIZE) as i32;
                    chunk_valid_ranges.push(((chunk.offset as i32) << 8) | len);
                    seq_chunk_map.push((seq_idx, chunk_idx));
                } else {
                    preserve_per_seq[seq_idx].push((chunk_idx, chunk.clone()));
                }
            }
        }
        Ok::<(), candle::Error>(())
    })??;
    if chunk_jobs.is_empty() {
        return Ok(None);
    }
    Ok(Some(QuantBucket {
        chunk_jobs,
        chunk_valid_ranges,
        seq_chunk_map,
        preserve_per_seq,
    }))
}

/// One layer's selected per-`(chunk, head, palette)` quant formats, outer
/// scales, and packed palette maps — the host-side product of the selection
/// kernel, ready to drive the alloc + convert-descriptor build.
#[cfg(feature = "cuda")]
pub(super) struct LayerFormats {
    pub(super) k_palette_formats: Vec<Vec<QuantFormat>>,
    pub(super) v_palette_formats: Vec<Vec<QuantFormat>>,
    pub(super) k_palette_scales: Vec<Vec<f32>>,
    pub(super) v_palette_scales: Vec<Vec<f32>>,
    pub(super) k_palette_maps: Vec<Vec<u8>>,
    pub(super) v_palette_maps: Vec<Vec<u8>>,
}

/// Reshape the selection kernel's flat per-`(chunk, head)` rows into the
/// per-chunk [`LayerFormats`]. The input rows may be a slice of a larger
/// cross-layer readback (this layer's chunk range); `n_chunks` is this layer's
/// chunk count. Pure host transform — identical whether the rows came from a
/// per-layer or a batched cross-layer selection.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn derive_layer_formats(
    k_palette4_rows: &[[SampleFormat; 4]],
    v_palette4_rows: &[[SampleFormat; 4]],
    k_palette_scale_rows: &[[f32; 4]],
    v_palette_scale_rows: &[[f32; 4]],
    k_assignments: &[u8],
    v_assignments: &[u8],
    n_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> Result<LayerFormats> {
    let to_quant = |fmt: SampleFormat| -> Result<QuantFormat> {
        fmt.to_quant_format().ok_or_else(|| {
            candle::Error::Msg(format!(
                "quantize_sealed_in_place: selection produced non-quant format {fmt}"
            ))
        })
    };
    let build_fmt = |rows: &[[SampleFormat; 4]]| -> Result<Vec<Vec<QuantFormat>>> {
        (0..n_chunks)
            .map(|c| {
                let start = c * n_kv_head;
                let end = start + n_kv_head;
                rows[start..end]
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .map(to_quant)
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()
    };
    let build_scale = |rows: &[[f32; 4]]| -> Vec<Vec<f32>> {
        (0..n_chunks)
            .map(|c| {
                let start = c * n_kv_head;
                let end = start + n_kv_head;
                rows[start..end]
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect::<Vec<f32>>()
            })
            .collect()
    };
    Ok(LayerFormats {
        k_palette_formats: build_fmt(k_palette4_rows)?,
        v_palette_formats: build_fmt(v_palette4_rows)?,
        k_palette_scales: build_scale(k_palette_scale_rows),
        v_palette_scales: build_scale(v_palette_scale_rows),
        k_palette_maps: pack_head_palette_maps(k_assignments, n_chunks, n_kv_head, head_dim)?,
        v_palette_maps: pack_head_palette_maps(v_assignments, n_chunks, n_kv_head, head_dim)?,
    })
}

/// Add to `demand` the destination slots, by class, that quantizing chunks of
/// these formats allocates: one per `(chunk, head, palette, side)` band, in its
/// format's class, the policy's uniform overrides applied. What a refused
/// allocation owes, read from the same formats the allocation reads.
#[cfg(feature = "cuda")]
pub(super) fn add_layer_demand(
    backing: &ChunkedKvBacking,
    policy: &CompressionPolicy,
    k_palette_formats: &[Vec<QuantFormat>],
    v_palette_formats: &[Vec<QuantFormat>],
    n_kv_head: usize,
    demand: &mut Vec<(ArenaKey, usize)>,
) -> Result<()> {
    for (k_chunk, v_chunk) in k_palette_formats.iter().zip(v_palette_formats) {
        for band in 0..n_kv_head * N_PALETTE {
            let k = policy.override_k_quant.unwrap_or(k_chunk[band]);
            let v = policy.override_v_quant.unwrap_or(v_chunk[band]);
            for format in [k, v] {
                let key = backing
                    .inner
                    .arena_key_for(KvFormat::Quantized(format), ArenaLocation::Gpu)?;
                match demand.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, n)) => *n += 1,
                    None => demand.push((key, 1)),
                }
            }
        }
    }
    Ok(())
}

/// Cross-layer batched quantize of sealed sequences: run the format-selection
/// kernel **once** across every layer's chunks (via
/// [`PagedSelectionGpuInputs::from_head_gids_multi`]) instead of once per
/// layer, then quantize each layer with its slice of the shared selection —
/// deferring the convert into `descs_acc` exactly like
/// [`quantize_sealed_in_place_deferred`]. `backings[i]` owns
/// `per_layer_seqs[i]`; returns the GPU-quant sequences per layer, in order,
/// each selection run cut to carve at most `selection_budget` bytes.
/// Bit-identical to the per-layer path: the unified table's per-layer
/// `arena_idx` offset is a pure relabelling of rows the kernel reads by
/// pointer, and chunks never interact across the selection. The reference
/// seal the gap quantize is held to, byte for byte, by the device tests.
#[cfg(all(test, feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_layers_deferred_within(
    backings: &[&ChunkedKvBacking],
    per_layer_seqs: &[Vec<&SealedSequence>],
    policy: &CompressionPolicy,
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    pinned_scratch: &mut Option<PinnedBuf>,
    descs_acc: &mut Vec<PalHeadDesc>,
    select_ms: &mut u64,
    alloc_ms: &mut u64,
    selection_budget: usize,
) -> Result<Vec<Vec<SealedSequence>>> {
    if backings.len() != per_layer_seqs.len() {
        candle::bail!("quantize_layers_deferred: backings/seqs length mismatch");
    }
    if backings.is_empty() {
        return Ok(Vec::new());
    }
    let cuda_dev = match device {
        Device::Cuda(d) => d,
        _ => candle::bail!("quantize_layers_deferred: requires a CUDA device"),
    };
    let n_kv_head = backings[0].n_kv_head();
    let head_dim = backings[0].head_dim();

    // Per-layer bucket. `None` ⇒ that layer has no kernel-eligible chunks.
    let buckets: Vec<Option<QuantBucket>> = backings
        .iter()
        .zip(per_layer_seqs)
        .map(|(b, seqs)| bucket_quant_chunks(b, seqs))
        .collect::<Result<Vec<_>>>()?;

    // Gather every layer's chunk_jobs + valid_ranges for ONE selection. Empty
    // entries stay in place so arena offsets and the per-layer split align with
    // `backings` indices.
    let backings_inner: Vec<_> = backings.iter().map(|b| b.inner.clone()).collect();
    let chunk_gids_per_layer: Vec<Vec<ChunkBands>> = buckets
        .iter()
        .map(|b| b.as_ref().map(|q| q.chunk_jobs.clone()).unwrap_or_default())
        .collect();
    let mut unified_valid_ranges: Vec<i32> = Vec::new();
    for b in buckets.iter().flatten() {
        unified_valid_ranges.extend_from_slice(&b.chunk_valid_ranges);
    }

    let total_jobs: usize = chunk_gids_per_layer.iter().map(|g| g.len()).sum();
    if total_jobs == 0 {
        // Nothing eligible anywhere — pass every layer through verbatim.
        return Ok(per_layer_seqs
            .iter()
            .map(|seqs| seqs.iter().map(|s| (*s).clone()).collect())
            .collect());
    }

    // Candidates + thresholds are policy-derived — identical for every layer.
    let PolicySelection {
        k: k_candidates,
        v: v_candidates,
        thresholds: [k_hi, k_lo, v_hi, v_lo],
    } = policy_selection(policy)?;

    // ── One fused selection per batch of chunks, across every layer ─────
    //
    // The selection carves its tables and its result block from the
    // persistence domain, which is `MIGRATION_STAGING_CAP_BYTES` wide. A group
    // of ingest documents is tens of thousands of chunks across the layers —
    // a quarter of a gigabyte of block — so one selection over all of it is
    // refused, every pass, and nothing ever leaves the GPU. So the flat
    // layer-major chunk list is cut into the longest runs that fit, each run
    // selected on its own. Chunks never interact across the selection, so the
    // concatenated results are the ones a single selection would return.
    let t_sel = std::time::Instant::now();
    let chunk_counts: Vec<usize> = chunk_gids_per_layer.iter().map(Vec::len).collect();
    let (n_k, n_v) = (k_candidates.len(), v_candidates.len());
    let batches = selection_batches(total_jobs, selection_budget, |c| {
        selection_bytes(c, n_kv_head, head_dim, n_k, n_v)
    })?;
    let (mut k_rows, mut v_rows) = (Vec::new(), Vec::new());
    let (mut k_scale_rows, mut v_scale_rows) = (Vec::new(), Vec::new());
    let (mut k_assign, mut v_assign) = (Vec::new(), Vec::new());
    let stager = backings[0].begin_stager_generation_required();
    for (start, len) in batches {
        let batch_gids = slice_layers(&chunk_gids_per_layer, start, len);
        let (gpu_inputs, _) = PagedSelectionGpuInputs::from_head_gids_multi(
            &backings_inner,
            &batch_gids,
            Some(&stager),
            cuda_dev,
        )?;
        let (kr, vr, ks, vs, ka, va, _, _) = gpu_inputs.select_palette4_formats_fused(
            &k_candidates,
            &v_candidates,
            k_hi,
            k_lo,
            v_hi,
            v_lo,
            Some(&unified_valid_ranges[start..start + len]),
            Some(&stager),
        )?;
        k_rows.extend(kr);
        v_rows.extend(vr);
        k_scale_rows.extend(ks);
        v_scale_rows.extend(vs);
        k_assign.extend(ka);
        v_assign.extend(va);
    }
    drop(stager);
    *select_ms += t_sel.elapsed().as_millis() as u64;

    // Split the flat rows per layer (chunk-major, then head; assignments also ×
    // head_dim) into each layer's formats, before any layer allocates: a layer
    // refused mid-wave owes what every layer still to allocate needs.
    let rows_per_chunk = n_kv_head;
    let assign_per_chunk = n_kv_head * head_dim;
    let mut chunk_base = 0usize;
    let mut plans: Vec<Option<(QuantBucket, LayerFormats)>> = Vec::with_capacity(backings.len());
    for (li, bucket) in buckets.into_iter().enumerate() {
        let Some(bucket) = bucket else {
            plans.push(None);
            continue;
        };
        let n_chunks = chunk_counts[li];
        debug_assert_eq!(n_chunks, bucket.chunk_jobs.len());
        let r0 = chunk_base * rows_per_chunk;
        let r1 = (chunk_base + n_chunks) * rows_per_chunk;
        let a0 = chunk_base * assign_per_chunk;
        let a1 = (chunk_base + n_chunks) * assign_per_chunk;
        let formats = derive_layer_formats(
            &k_rows[r0..r1],
            &v_rows[r0..r1],
            &k_scale_rows[r0..r1],
            &v_scale_rows[r0..r1],
            &k_assign[a0..a1],
            &v_assign[a0..a1],
            n_chunks,
            n_kv_head,
            head_dim,
        )?;
        chunk_base += n_chunks;
        plans.push(Some((bucket, formats)));
    }

    // Finish each layer, deferring its convert into `descs_acc`.
    let mut out: Vec<Vec<SealedSequence>> = Vec::with_capacity(backings.len());
    for li in 0..plans.len() {
        let seqs = &per_layer_seqs[li];
        let Some(plan) = plans[li].take() else {
            out.push(seqs.iter().map(|s| (*s).clone()).collect());
            continue;
        };
        let mut owed_if_refused: Vec<(ArenaKey, usize)> = Vec::new();
        add_layer_demand(
            backings[li],
            policy,
            &plan.1.k_palette_formats,
            &plan.1.v_palette_formats,
            n_kv_head,
            &mut owed_if_refused,
        )?;
        match quantize_sealed_in_place_impl(
            backings[li],
            seqs,
            policy,
            device,
            copy_stream,
            pinned_scratch,
            Some(&mut *descs_acc),
            Some(plan),
            Some(&mut *alloc_ms),
        ) {
            Ok(layer) => out.push(layer),
            // **Refused mid-wave: owe the rest of the group, not this layer.**
            // The layers share one pool per class. A group refused at one layer
            // and owed only that layer's arenas got past it at the retry and was
            // refused at the next — thirteen layers, thirteen gaps, while the
            // ingest's own writes took each gap's arenas. Measured on BlackWorld
            // (2026-11-10): one residence short ~4,200 R16 slots a layer, every
            // retry, for forty minutes. This owes the refused layer's demand
            // together with every layer after it; `owe_capacity` counts the
            // arenas the refused layer already owed, so the gap makes the rest
            // at once. The layers before it free what they claimed as the group
            // drops, and their retry claims it back.
            Err(e) if super::super::backing::BackingInner::is_wave_deferral(&e) => {
                for (_, formats) in plans[li + 1..].iter().flatten() {
                    add_layer_demand(
                        backings[li],
                        policy,
                        &formats.k_palette_formats,
                        &formats.v_palette_formats,
                        n_kv_head,
                        &mut owed_if_refused,
                    )?;
                }
                backings[li].inner.owe_capacity(&owed_if_refused);
                return Err(e);
            }
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}
