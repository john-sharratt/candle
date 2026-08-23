//! Quantize-on-evict — adaptive per-`(head, palette)` compression at the
//! hot→warm tier boundary.
//!
//! The substrate's persistence thread calls [`quantize_sealed_in_place`]
//! when a `CompressionPolicy` is configured. The flow within this module is:
//!
//! ```text
//!   GPU Float / R16  ──select_palette4_formats_fused──▶  per-(h, p) format choice
//!                    ──quantize_palette4_convert_buffered──▶  GPU Quant arenas
//! ```
//!
//! The selection kernel reads the source float bytes and picks a
//! quantization format per `(chunk, head, palette)` from the policy's
//! candidate list using the production K/V error thresholds. The
//! conversion kernel writes the quantized bytes into GPU-quant arena
//! slots. The returned `SealedSequence`s still live in VRAM — the
//! persist thread does a separate format-preserving DtoH copy via
//! `migrate_sealed_to_cpu_batch_async` to produce the warm-tier image,
//! and then `install_warm_and_hot` atomically replaces hot with the new
//! Q-format GPU chunks and installs warm.
//!
//! Source-side rules:
//! - Eligible chunks must have all their (h, p) GIDs in GPU `Float` or
//!   `Quantized(R16)` arenas — the formats the selection + convert
//!   kernels read directly. Ineligible chunks (typically cold-loaded
//!   ones already in mixed Q-formats) are routed to a pass-through
//!   "preserve" bucket and merged back unchanged at the end.
//! - `head_dim` must be 128 — the palette-4 kernel is compiled for
//!   that shape only.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::CudaStream;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::{dtype_to_ggml_float, identity_pal_map, PalHeadDesc};
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::{Generation, PinnedBuf};
#[cfg(feature = "cuda")]
use candle::quantized::GgmlDType;
#[cfg(feature = "cuda")]
use candle::{DType, Device, Result};

#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
use super::backing::ChunkedKvBacking;
#[cfg(feature = "cuda")]
use super::compression_policy::{
    CompressionPolicy, PRODUCTION_K_QREL_HIGH_THRESHOLDS, PRODUCTION_K_QREL_LOW_THRESHOLDS,
    PRODUCTION_V_QREL_HIGH_THRESHOLDS, PRODUCTION_V_QREL_LOW_THRESHOLDS,
};
#[cfg(feature = "cuda")]
use super::gid_pool::ChunkGid;
#[cfg(feature = "cuda")]
use super::head_gids::{ChunkBands, HeadGids, GIDS_PER_HEAD};
#[cfg(feature = "cuda")]
use super::meta_pool::ChunkRecordSrc;
#[cfg(feature = "cuda")]
use super::sampled_selection::{PagedSelectionGpuInputs, SampleFormat};
#[cfg(feature = "cuda")]
use super::types::{SealedChunk, SealedSequence, CHUNK_SIZE};
#[cfg(feature = "cuda")]
use crate::kv_cache::arena_table::{ArenaFormatTag, ArenaLocation, N_PALETTE};
#[cfg(feature = "cuda")]
use crate::kv_cache::{KvFormat, QuantFormat};

/// Device pointer of one band's chunk slot.
///
/// Every pointer the convert kernels take is this: a slot *is* one
/// `(head, palette, side)` band, so its address is the arena's base plus
/// `chunk_idx * class_stride` and nothing else. Under per-format arenas this
/// had to be spelled two ways — `tensor_ptr_at_offset` scaled an element
/// offset by a dtype width, `qtensor_ptr_at_byte_offset` took a byte offset
/// computed from a ggml block layout — and both were reconstructing an address
/// the arena can hand over directly.
#[cfg(feature = "cuda")]
fn band_ptr(storage: &super::arena::ArenaStorageState, gid: &ChunkGid, what: &str) -> Result<u64> {
    let ai = gid.arena_idx();
    let arena = storage
        .arenas()
        .get(&ai)
        .ok_or_else(|| candle::Error::Msg(format!("{what} arena {ai} not found")))?;
    arena
        .slot_ptr(gid.chunk_idx())
        .ok_or_else(|| candle::Error::Msg(format!("{what} arena {ai} is not GPU-resident")))
}

/// The GGML layout a band's bytes are in, from the band's own tag.
///
/// The arena cannot answer this — it holds whatever fits its slots
/// (`docs/archived/arena_unification.md` principle 8) — so the tag is the only source,
/// and it is the same byte the substrate persisted.
#[cfg(feature = "cuda")]
fn band_ggml_dtype(tag: u8) -> Result<GgmlDType> {
    let format = KvFormat::from_tag(tag).ok_or_else(|| {
        candle::Error::Msg(format!("chunk band carries unrecognised format tag {tag}"))
    })?;
    match format {
        KvFormat::Quantized(qf) => Ok(qf.to_ggml_dtype()),
        KvFormat::Float(dt) => dtype_to_ggml_float(dt),
    }
}

/// Pack per-dim palette assignments (one byte per dim, value in `{0..3}`)
/// into the 32-byte packed map the decode/prefill kernels read.
///
/// Layout in `assignments`: `[chunk][head][dim]`, row-major.
/// Layout in output `head_palette_maps`: one `Vec<u8>` per chunk, each of
/// length `n_head * (head_dim/4)` bytes. Within each head's packed slice,
/// dim `d` occupies bits `2*(d%4) .. 2*(d%4)+2` of byte `d/4`.
///
/// Enforces the palette-4 invariant: each of the 4 palette slots must
/// receive exactly `head_dim / N_PALETTE` dimensions; the kernel's
/// per-palette quantization assumes that strict partition.
#[cfg(feature = "cuda")]
fn pack_head_palette_maps(
    assignments: &[u8],
    n_chunks: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<Vec<Vec<u8>>> {
    let packed_bytes = head_dim / 4;
    if !head_dim.is_multiple_of(4)
        || !head_dim.is_multiple_of(N_PALETTE)
        || !candle::quantized::cuda::kvhead_supported_head_dim(head_dim)
    {
        candle::bail!(
            "quantize_sealed_to_cpu: palette4 conversion requires head_dim 128 or 256, got {}",
            head_dim,
        );
    }

    let expected = n_chunks
        .checked_mul(n_head)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| candle::Error::Msg("palette map shape overflow".into()))?;
    if assignments.len() != expected {
        candle::bail!(
            "palette map length mismatch: got {}, expected {} ({} dims/head × {} chunks × {} heads)",
            assignments.len(),
            expected,
            head_dim,
            n_chunks,
            n_head,
        );
    }

    let target_per_palette = head_dim / N_PALETTE;
    let mut out = Vec::with_capacity(n_chunks);
    for chunk_i in 0..n_chunks {
        let mut block_map = vec![0u8; n_head * packed_bytes];
        for h in 0..n_head {
            let base = (chunk_i * n_head + h) * head_dim;
            let mut counts = [0usize; N_PALETTE];
            for (dim_offset, &raw_slot) in assignments[base..base + head_dim].iter().enumerate() {
                let slot = (raw_slot & 0x3) as usize;
                counts[slot] += 1;
                let byte_idx = h * packed_bytes + (dim_offset / 4);
                block_map[byte_idx] |= (slot as u8) << (2 * (dim_offset % 4));
            }
            if counts.iter().any(|&count| count != target_per_palette) {
                candle::bail!(
                    "invalid palette4 routing for chunk {} head {}: {:?} (each palette must \
                     receive exactly {} dims)",
                    chunk_i,
                    h,
                    counts,
                    target_per_palette,
                );
            }
        }
        out.push(block_map);
    }
    Ok(out)
}

/// Adaptive per-(head, palette) quantization that runs entirely on GPU
/// and returns `SealedSequence`s whose chunks still live in VRAM.
///
/// Eligible chunks land in fresh GPU Q-format arenas chosen by the
/// palette-4 selection kernel. Ineligible chunks (those whose source
/// GIDs are not in GPU `Float` / `Quantized(R16)` arenas — typically
/// cold-loaded chunks already in mixed Q-formats) flow through a
/// pass-through "preserve" bucket and get merged back unchanged in
/// the output sequence, preserving original chunk order.
///
/// **No DMA to CPU.** Callers that need warm-tier storage feed the
/// returned GPU sequences to [`ChunkedKvBacking::migrate_sealed_to_cpu_batch_async`]
/// for a separate, format-preserving DtoH copy. The split keeps the
/// in-session reconcile purely on-GPU (decode reads the convert
/// kernel's output bytes directly, no intervening `kv_migrate` scatter).
#[cfg(feature = "cuda")]
/// Single-layer entry point: quantize + launch the convert immediately. See
/// [`quantize_sealed_in_place_deferred`] for the cross-layer batched form.
pub fn quantize_sealed_in_place(
    backing: &ChunkedKvBacking,
    sequences: &[&SealedSequence],
    policy: &CompressionPolicy,
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    pinned_scratch: &mut Option<PinnedBuf>,
) -> Result<Vec<SealedSequence>> {
    quantize_sealed_in_place_impl(
        backing,
        sequences,
        policy,
        device,
        copy_stream,
        pinned_scratch,
        None,
        None,
        None,
    )
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

/// Cross-layer batched quantize: run the format-selection kernel **once** across
/// every layer's chunks (via [`PagedSelectionGpuInputs::from_head_gids_multi`])
/// instead of once per layer, then quantize each layer with its slice of the
/// shared selection — deferring the convert into `descs_acc` exactly like
/// [`quantize_sealed_in_place_deferred`]. This collapses the per-layer selection
/// launch + 8 host readbacks (the WDDM-bound term that dominates the hot→warm
/// drain) to a single launch + readback. `backings[i]` owns `per_layer_seqs[i]`;
/// returns the GPU-quant sequences per layer, in order. Bit-identical to the
/// per-layer path: the unified table's per-layer `arena_idx` offset is a pure
/// relabelling of rows the kernel reads by pointer, and chunks never interact
/// across the selection.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn quantize_layers_deferred(
    backings: &[&ChunkedKvBacking],
    per_layer_seqs: &[Vec<&SealedSequence>],
    policy: &CompressionPolicy,
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    pinned_scratch: &mut Option<PinnedBuf>,
    descs_acc: &mut Vec<PalHeadDesc>,
    // Timing breakdown of this pass's quantize phase, so the persist log can see
    // whether the cross-layer selection or the dst-arena allocation dominates.
    select_ms: &mut u64,
    alloc_ms: &mut u64,
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
    let (k_candidates, v_candidates) = {
        let (k_profile, v_profile) =
            super::compression_policy::production_adaptive_candidates(policy.compression_level);
        let k: Vec<SampleFormat> = k_profile
            .into_iter()
            .filter_map(SampleFormat::from_kv_format)
            .filter(|f| !f.is_float())
            .collect();
        let v: Vec<SampleFormat> = v_profile
            .into_iter()
            .filter_map(SampleFormat::from_kv_format)
            .filter(|f| !f.is_float())
            .collect();
        if k.is_empty() || v.is_empty() {
            candle::bail!(
                "quantize_layers_deferred: missing K/V production candidates for level {}",
                policy.compression_level
            );
        }
        (k, v)
    };
    let ti = policy.compression_level.min(10) as usize;
    let k_hi = PRODUCTION_K_QREL_HIGH_THRESHOLDS[ti] * policy.k_hi_error_threshold_factor;
    let k_lo = PRODUCTION_K_QREL_LOW_THRESHOLDS[ti] * policy.k_low_error_threshold_factor;
    let v_hi = PRODUCTION_V_QREL_HIGH_THRESHOLDS[ti] * policy.v_hi_error_threshold_factor;
    let v_lo = PRODUCTION_V_QREL_LOW_THRESHOLDS[ti] * policy.v_low_error_threshold_factor;

    // ── ONE selection across every layer's chunks ──────────────────────
    let t_sel = std::time::Instant::now();
    let stager = backings[0].begin_stager_generation_required();
    let (gpu_inputs, chunk_counts) = PagedSelectionGpuInputs::from_head_gids_multi(
        &backings_inner,
        &chunk_gids_per_layer,
        Some(&stager),
        cuda_dev,
    )?;
    let (k_rows, v_rows, k_scale_rows, v_scale_rows, k_assign, v_assign, _ka, _va) = gpu_inputs
        .select_palette4_formats_fused(
            &k_candidates,
            &v_candidates,
            k_hi,
            k_lo,
            v_hi,
            v_lo,
            Some(&unified_valid_ranges),
            Some(&stager),
        )?;
    drop(gpu_inputs);
    drop(stager);
    *select_ms += t_sel.elapsed().as_millis() as u64;

    // Split the flat rows per layer (chunk-major, then head; assignments also ×
    // head_dim) and finish each layer, deferring its convert into `descs_acc`.
    let rows_per_chunk = n_kv_head;
    let assign_per_chunk = n_kv_head * head_dim;
    let mut out: Vec<Vec<SealedSequence>> = Vec::with_capacity(backings.len());
    let mut chunk_base = 0usize;
    for (li, bucket) in buckets.into_iter().enumerate() {
        let seqs = &per_layer_seqs[li];
        let Some(bucket) = bucket else {
            out.push(seqs.iter().map(|s| (*s).clone()).collect());
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
        out.push(quantize_sealed_in_place_impl(
            backings[li],
            seqs,
            policy,
            device,
            copy_stream,
            pinned_scratch,
            Some(descs_acc),
            Some((bucket, formats)),
            Some(&mut *alloc_ms),
        )?);
    }
    Ok(out)
}

/// One layer's chunks partitioned for quantization: the kernel-eligible jobs
/// (full/partial GPU-`Float` or `R16` chunks) and the preserve bucket (mixed-Q
/// chunks passed through unchanged). Owned, so it outlives the storage read lock
/// and can be threaded through a cross-layer selection.
#[cfg(feature = "cuda")]
struct QuantBucket {
    /// One entry per kernel-eligible chunk: its band gids **and** the band
    /// format tags that say how to read them. The selection table and the
    /// convert-descriptor build both need the pair; the arena a gid points into
    /// can only answer the first half.
    chunk_jobs: Vec<ChunkBands>,
    chunk_valid_ranges: Vec<i32>,
    seq_chunk_map: Vec<(usize, usize)>,
    preserve_per_seq: Vec<Vec<(usize, SealedChunk)>>,
}

/// Partition `sequences`' chunks into kernel-eligible jobs vs the preserve
/// bucket. A chunk is eligible when every `(h, p)` source band is GPU-resident
/// and recorded as `Float` or `Quantized(R16)` — the formats the selection +
/// convert kernels read directly. Location comes from the arena (it is arena
/// identity); the format comes from the chunk's own band tags, because under
/// size classes an arena holds whatever fits its stride and cannot answer the
/// question. Full and partial chunks both qualify (partial dead
/// slots are zero — arenas are zeroed at creation/recycle — and the packed
/// valid range corrects the count-normalized metrics). Ineligible chunks (e.g. a
/// view borrowing cold-loaded mixed-Q chunks) go to `preserve` and merge back
/// unchanged. `Ok(None)` ⇒ nothing eligible (caller passes the sequences
/// through verbatim).
#[cfg(feature = "cuda")]
fn bucket_quant_chunks(
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
                    let ok = super::chunk_ops::needs_reconcile_source_tag(tag)
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
struct LayerFormats {
    k_palette_formats: Vec<Vec<QuantFormat>>,
    v_palette_formats: Vec<Vec<QuantFormat>>,
    k_palette_scales: Vec<Vec<f32>>,
    v_palette_scales: Vec<Vec<f32>>,
    k_palette_maps: Vec<Vec<u8>>,
    v_palette_maps: Vec<Vec<u8>>,
}

/// Reshape the selection kernel's flat per-`(chunk, head)` rows into the
/// per-chunk [`LayerFormats`]. The input rows may be a slice of a larger
/// cross-layer readback (this layer's chunk range); `n_chunks` is this layer's
/// chunk count. Pure host transform — identical whether the rows came from a
/// per-layer or a batched cross-layer selection.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn derive_layer_formats(
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

fn quantize_sealed_in_place_impl(
    backing: &ChunkedKvBacking,
    sequences: &[&SealedSequence],
    policy: &CompressionPolicy,
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    pinned_scratch: &mut Option<PinnedBuf>,
    deferred_descs: Option<&mut Vec<PalHeadDesc>>,
    precomputed: Option<(QuantBucket, LayerFormats)>,
    // Accumulator for the dst-arena allocation time (the suspected
    // pressure-regime bottleneck). `None` on single-layer callers.
    alloc_ms: Option<&mut u64>,
) -> Result<Vec<SealedSequence>> {
    let _ = pinned_scratch; // kept for signature compatibility; no DtoH performed.
    if sequences.is_empty() {
        return Ok(Vec::new());
    }

    // This is a compress-to-free op: it writes small quantized chunks, then
    // frees the much larger float source. That used to need a byte reserve only
    // it could draw on, or under extreme pressure the compress would fail for
    // want of memory and the cache could never reclaim any (freeing needs
    // memory). Scarcity-only class promotion is the answer now: with no free
    // region, the small chunks take free slots in a wider class's existing
    // region, and the float regions they release come back whole
    // (`docs/archived/arena_unification.md` §3.4).

    let cuda_dev = match device {
        Device::Cuda(d) => d,
        _ => candle::bail!("quantize_sealed_in_place: requires a CUDA device"),
    };

    let n_kv_head = backing.n_kv_head();
    let head_dim = backing.head_dim();
    if backing.single_latent() {
        candle::bail!("single-latent KV is not adaptively compressed");
    }
    if !candle::quantized::cuda::kvhead_supported_head_dim(head_dim) {
        candle::bail!(
            "quantize_sealed_in_place: palette4 conversion requires head_dim 128 or 256, \
             got {head_dim}"
        );
    }
    let sub_head_dim = head_dim / N_PALETTE;
    let elems_per_head = CHUNK_SIZE * sub_head_dim;

    // ── Bucket each source chunk: quantize vs preserve ──────────────
    //
    // A chunk is eligible for the fused palette4 kernel when every
    // (h, p) source GID lives in a GPU `Float` or `Quantized(R16)`
    // arena — the formats the selection + convert kernels read
    // directly. Full and partial chunks both qualify; see the inner
    // loop for the partial-tail correctness argument.
    //
    // Ineligible chunks go into the `preserve` bucket and get merged
    // back into the output sequence unchanged at the end. The case
    // that hits production: a turn's view borrows older chunks that
    // came back from cold storage in mixed Q-formats. Those GIDs
    // already carry the format/pal_map/scale state an earlier persist
    // pass selected; re-quantizing would be redundant and the
    // source-format check would reject them anyway.
    //
    // `chunk_jobs[i]` corresponds to one SealedChunk that will go
    // through the kernel. `seq_chunk_map[i] = (seq_idx, chunk_idx)`
    // lets us reassemble per-sequence outputs in the original chunk
    // order at the end.
    // Bucket source chunks (quantize vs preserve) — from the caller's
    // precomputed cross-layer plan when present, else a fresh per-layer bucket.
    let (bucket, precomputed_formats) = match precomputed {
        Some((b, f)) => (b, Some(f)),
        None => match bucket_quant_chunks(backing, sequences)? {
            Some(b) => (b, None),
            None => return Ok(sequences.iter().map(|s| (*s).clone()).collect()),
        },
    };
    let QuantBucket {
        chunk_jobs,
        chunk_valid_ranges,
        seq_chunk_map,
        preserve_per_seq,
    } = bucket;
    let n_chunks = chunk_jobs.len();

    // ── Per-(chunk, head, palette) format selection ───────────────────
    // Use the caller's precomputed cross-layer selection when present; else run
    // this layer's selection kernel now (one launch across all chunk jobs) and
    // reshape its flat readback rows. `stager_generation` is `Some` only on the
    // per-layer path — the deferred cross-layer path never runs an immediate
    // convert, so it needs no stager here.
    let (
        LayerFormats {
            k_palette_formats,
            v_palette_formats,
            k_palette_scales,
            v_palette_scales,
            k_palette_maps,
            v_palette_maps,
        },
        stager_generation,
    ) = match precomputed_formats {
        Some(f) => (f, None),
        None => {
            let (k_candidates, v_candidates) = {
                let (k_profile, v_profile) =
                    super::compression_policy::production_adaptive_candidates(
                        policy.compression_level,
                    );
                let k: Vec<SampleFormat> = k_profile
                    .into_iter()
                    .filter_map(SampleFormat::from_kv_format)
                    .filter(|fmt| !fmt.is_float())
                    .collect();
                let v: Vec<SampleFormat> = v_profile
                    .into_iter()
                    .filter_map(SampleFormat::from_kv_format)
                    .filter(|fmt| !fmt.is_float())
                    .collect();
                if k.is_empty() || v.is_empty() {
                    candle::bail!(
                        "quantize_sealed_in_place: missing K/V production candidates for level {}",
                        policy.compression_level
                    );
                }
                (k, v)
            };

            let threshold_idx = policy.compression_level.min(10) as usize;
            let k_threshold_hi = PRODUCTION_K_QREL_HIGH_THRESHOLDS[threshold_idx]
                * policy.k_hi_error_threshold_factor;
            let k_threshold_lo = PRODUCTION_K_QREL_LOW_THRESHOLDS[threshold_idx]
                * policy.k_low_error_threshold_factor;
            let v_threshold_hi = PRODUCTION_V_QREL_HIGH_THRESHOLDS[threshold_idx]
                * policy.v_hi_error_threshold_factor;
            let v_threshold_lo = PRODUCTION_V_QREL_LOW_THRESHOLDS[threshold_idx]
                * policy.v_low_error_threshold_factor;

            let sg = backing.begin_stager_generation_required();
            let gpu_inputs = PagedSelectionGpuInputs::from_head_gids(
                backing.inner.clone(),
                &chunk_jobs,
                Some(&sg),
                cuda_dev,
            )?;

            let (
                k_palette4_rows,
                v_palette4_rows,
                k_palette_scale_rows,
                v_palette_scale_rows,
                k_assignments,
                v_assignments,
                _k_head_amax,
                _v_head_amax,
            ) = gpu_inputs.select_palette4_formats_fused(
                &k_candidates,
                &v_candidates,
                k_threshold_hi,
                k_threshold_lo,
                v_threshold_hi,
                v_threshold_lo,
                Some(&chunk_valid_ranges),
                Some(&sg),
            )?;

            let formats = derive_layer_formats(
                &k_palette4_rows,
                &v_palette4_rows,
                &k_palette_scale_rows,
                &v_palette_scale_rows,
                &k_assignments,
                &v_assignments,
                n_chunks,
                n_kv_head,
                head_dim,
            )?;
            (formats, Some(sg))
        }
    };

    // ── Allocate GPU-quant destination GIDs ───────────────────────────
    // One destination GID per (chunk, head, palette, K/V). The
    // candidate arenas are warm-protected at backing creation when
    // the policy is set, so `alloc_chunk_for_key` reuses them.
    //
    // When `policy.override_k_quant` is set, K dst is uniform `fmt`
    // regardless of selection's K format pick. V always uses selection's
    // per-(c, h, p) format.
    let t_alloc = std::time::Instant::now();
    let mut new_gids_per_chunk: Vec<Vec<ChunkGid>> = Vec::with_capacity(n_chunks);
    for chunk_i in 0..n_chunks {
        // Per (head, side): resolve the N_PALETTE band formats up front. A
        // uniform group allocates one CONTIGUOUS run for select/QREL walk
        // locality (correctness is per-band via `resolve_band_source`, not the
        // run layout — see `alloc_chunk_run_for_key`); mixed-format groups
        // allocate per band.
        let mut k_gids: Vec<Vec<ChunkGid>> = Vec::with_capacity(n_kv_head);
        let mut v_gids: Vec<Vec<ChunkGid>> = Vec::with_capacity(n_kv_head);
        for h in 0..n_kv_head {
            let k_fmts: Vec<_> = (0..N_PALETTE)
                .map(|p| {
                    policy
                        .override_k_quant
                        .unwrap_or(k_palette_formats[chunk_i][h * N_PALETTE + p])
                })
                .collect();
            let v_fmts: Vec<_> = (0..N_PALETTE)
                .map(|p| {
                    policy
                        .override_v_quant
                        .unwrap_or(v_palette_formats[chunk_i][h * N_PALETTE + p])
                })
                .collect();
            // Eligibility for one contiguous run is a question about *keys*:
            // four bands in four different formats that share a size class
            // share an arena, so they can still be a run. Under per-format
            // arenas this could only pass for identical formats.
            let alloc_side = |fmts: &[crate::kv_cache::QuantFormat]| -> Result<Vec<ChunkGid>> {
                let keys = fmts
                    .iter()
                    .map(|f| {
                        backing
                            .inner
                            .arena_key_for(KvFormat::Quantized(*f), ArenaLocation::Gpu)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if keys.iter().all(|k| *k == keys[0]) {
                    backing.alloc_chunk_run_for_key(keys[0], N_PALETTE)
                } else {
                    keys.into_iter()
                        .map(|k| backing.alloc_chunk_for_key(k))
                        .collect()
                }
            };
            k_gids.push(alloc_side(&k_fmts)?);
            v_gids.push(alloc_side(&v_fmts)?);
        }
        let mut chunk_gids: Vec<ChunkGid> = Vec::with_capacity(n_kv_head * GIDS_PER_HEAD);
        for h in 0..n_kv_head {
            for p in 0..N_PALETTE {
                chunk_gids.push(k_gids[h][p].clone());
                chunk_gids.push(v_gids[h][p].clone());
            }
        }
        new_gids_per_chunk.push(chunk_gids);
    }
    if let Some(a) = alloc_ms {
        *a += t_alloc.elapsed().as_millis() as u64;
    }

    // ── Build PalHeadDesc structs + launch kernel ─────────────────────
    let mut descs: Vec<PalHeadDesc> = Vec::with_capacity(n_chunks * n_kv_head);

    backing.inner.storage.try_write(|storage| {
        for (chunk_i, src_bands) in chunk_jobs.iter().enumerate() {
            let src_gids = &src_bands.gids;
            let ident = identity_pal_map(head_dim);
            // The source chunk's own band tags. `bucket_quant_chunks` already
            // proved every band is Float or R16, so the only question left per
            // band is *which* of the two.
            let src_bands = &chunk_jobs[chunk_i];
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

                    // K source — layout from the chunk's tag, address from the
                    // arena. `bucket_quant_chunks` already proved every band is
                    // Float or R16.
                    let k_src = src_gids.k_gid_pal(h, p);
                    k_src_fmts[p] = band_ggml_dtype(src_bands.band_tags(h, p).0)?;
                    k_src_ptrs[p] = band_ptr(storage, k_src, "k src")?;

                    // K destination — `policy.override_k_quant` decides
                    // between uniform override format and selection's
                    // per-(chunk, head, palette) pick. Must match the K
                    // arena alloc above.
                    let k_dst = &new_gids_per_chunk[chunk_i][gid_slot];
                    let k_qfmt = policy
                        .override_k_quant
                        .unwrap_or(k_palette_formats[chunk_i][h * N_PALETTE + p]);
                    let k_qdtype = k_qfmt.to_ggml_dtype();
                    k_dst_fmts[p] = k_qdtype;
                    k_dst_ptrs[p] = band_ptr(storage, k_dst, "k dst")?;

                    // V source — same shape as the K side above.
                    let v_src = src_gids.v_gid_pal(h, p);
                    v_src_fmts[p] = band_ggml_dtype(src_bands.band_tags(h, p).1)?;
                    v_src_ptrs[p] = band_ptr(storage, v_src, "v src")?;

                    // V destination — `policy.override_v_quant` decides
                    // between uniform override format and selection's
                    // per-(chunk, head, palette) pick. Must match the V
                    // arena alloc above.
                    let v_dst = &new_gids_per_chunk[chunk_i][gid_slot + 1];
                    let v_qfmt = policy
                        .override_v_quant
                        .unwrap_or(v_palette_formats[chunk_i][h * N_PALETTE + p]);
                    let v_qdtype = v_qfmt.to_ggml_dtype();
                    v_dst_fmts[p] = v_qdtype;
                    v_dst_ptrs[p] = band_ptr(storage, v_dst, "v dst")?;
                }

                // Only the first head_dim/4 bytes of the (max-width) map are
                // live; the per-head slices in *_palette_maps are that wide.
                let pal_bytes = head_dim / 4;
                let pal_start = h * pal_bytes;
                let pal_end = pal_start + pal_bytes;
                // K and V dst pal_maps: identity when the corresponding
                // override is active, else selection's per-(chunk, head).
                let k_dst_pal_map = if policy.override_k_quant.is_some() {
                    ident
                } else {
                    let mut m = ident;
                    m[..pal_bytes].copy_from_slice(&k_palette_maps[chunk_i][pal_start..pal_end]);
                    m
                };
                let v_dst_pal_map = if policy.override_v_quant.is_some() {
                    ident
                } else {
                    let mut m = ident;
                    m[..pal_bytes].copy_from_slice(&v_palette_maps[chunk_i][pal_start..pal_end]);
                    m
                };

                let scale_start = h * N_PALETTE;
                let scale_end = scale_start + N_PALETTE;
                // K and V dst scales: unit (1.0) when the corresponding
                // override is active, else selection's per-palette.
                let k_dst_scales = if policy.override_k_quant.is_some() {
                    [1.0f32; N_PALETTE]
                } else {
                    let mut s = [1.0f32; N_PALETTE];
                    s.copy_from_slice(&k_palette_scales[chunk_i][scale_start..scale_end]);
                    s
                };
                let v_dst_scales = if policy.override_v_quant.is_some() {
                    [1.0f32; N_PALETTE]
                } else {
                    let mut s = [1.0f32; N_PALETTE];
                    s.copy_from_slice(&v_palette_scales[chunk_i][scale_start..scale_end]);
                    s
                };

                descs.push(PalHeadDesc {
                    k_src_arena_ptrs: k_src_ptrs,
                    v_src_arena_ptrs: v_src_ptrs,
                    k_src_fmts,
                    v_src_fmts,
                    k_src_pal_map: ident,
                    v_src_pal_map: ident,
                    // Source is restricted to GPU Float (F16/F32/BF16) or
                    // Quantized(R16) by the eligibility check upstream; the
                    // palette-4 kernel API takes 1.0 for any float source
                    // because those formats have no outer-scale concept.
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
        }
        Ok(())
    })?;

    // All kernel work runs on the primary stream — selection, alloc, and
    // the convert kernel are FIFO on a single stream, so no cross-stream
    // fence is needed. The caller is responsible for syncing the primary
    // stream before the produced GPU bytes are consumed (the persist
    // thread does this once after the per-layer loop completes; that
    // single sync also covers any subsequent DtoH on `copy_stream`,
    // which records a wait on the primary stream's event).
    // Convert step. `deferred_descs = None` (single-layer callers) launches the
    // batched convert now; `Some(acc)` (the cross-layer persist pass) appends our
    // per-(chunk, head) descriptors to the caller's accumulator so it runs ONE
    // convert over EVERY layer's descriptors (collapsing the per-layer launches
    // too). Either way each block runs the unchanged kernel body against the same
    // descriptor — only the launch grouping changes, so it is bit-identical
    // (proven by the A/B tests). The dst arenas are already allocated above (the
    // reassembled sequences below point at them); the immediate-or-deferred
    // convert fills them on the primary stream before any downstream read.
    match deferred_descs {
        Some(acc) => acc.append(&mut descs),
        None => {
            let primary_stream = cuda_dev.cuda_stream();
            let sg = stager_generation.as_ref().ok_or_else(|| {
                candle::Error::Msg(
                    "quantize_sealed_in_place: immediate convert requires a stager (precomputed \
                     formats must defer the convert)"
                        .into(),
                )
            })?;
            convert_descs_batched(head_dim, &descs, n_kv_head, sg, &primary_stream)?;
        }
    }
    let _ = copy_stream; // intentionally unused: convert runs on primary.
    drop(stager_generation);

    // Convert-2 (the legacy "K → uniform Q8_0 + identity layout + unit scale"
    // re-quantize pass) has been removed. Its job is now done by
    // `policy.override_k_quant` at convert-1 time when the caller opts in,
    // and `None` lets selection's full K state propagate to storage directly.

    // ── Reassemble per-sequence GPU-quant SealedSequences ────────────
    // Only the dst quant arenas we just allocated are read below (via
    // `arena_byte_size` + `build_meta_records`), so resolve just those instead
    // of every arena — the O(num_arenas) `to_arena_entry` walk is the dominant
    // `other` term at pressure.
    let needed_arenas: std::collections::HashSet<usize> = new_gids_per_chunk
        .iter()
        .flat_map(|gids| gids.iter().map(|g| g.arena_idx()))
        .collect();
    let arena_infos = backing.resolve_arena_info_for(&needed_arenas)?;
    // For each (seq_idx, chunk_idx_within_seq), build the new GPU
    // Q-format SealedChunk. The kernel just wrote the per-(h, p)
    // palette4 bytes into the GIDs in `new_gids_per_chunk`; we wrap
    // those plus the policy-selected palette maps + outer scales (or
    // identity + 1.0 fallbacks when the corresponding override is set).
    let mut full_quant_chunks: std::collections::HashMap<(usize, usize), SealedChunk> =
        std::collections::HashMap::with_capacity(seq_chunk_map.len());
    // Keys in build order, so the batched record build below can map each
    // returned handle back to its chunk.
    let mut quant_keys: Vec<(usize, usize)> = Vec::with_capacity(seq_chunk_map.len());
    for (job_idx, &(seq_idx, chunk_idx)) in seq_chunk_map.iter().enumerate() {
        let new_gids = HeadGids::from_vec(new_gids_per_chunk[job_idx].clone());
        let (src_seq_idx, src_chunk_idx) = seq_chunk_map[job_idx];
        let src = &sequences[src_seq_idx].chunks[src_chunk_idx];
        // K and V pal_map / scale match what convert-1 wrote per side:
        // identity layout + empty Arc (1.0 fallback) when the corresponding
        // override is set; selection's full state otherwise. Identity bytes
        // are written explicitly per head because `gpu_chunks::rebuild_decode`
        // indexes them directly without a length-fallback check (unlike
        // `slot_state::from_sealed_chunk`).
        let pal_bytes_per_head = head_dim / 4;
        let identity_head_bytes: Option<Vec<u8>> =
            if policy.override_k_quant.is_some() || policy.override_v_quant.is_some() {
                let identity_head = identity_pal_map(head_dim);
                let mut buf = vec![0u8; n_kv_head * pal_bytes_per_head];
                for h in 0..n_kv_head {
                    buf[h * pal_bytes_per_head..(h + 1) * pal_bytes_per_head]
                        .copy_from_slice(&identity_head[..pal_bytes_per_head]);
                }
                Some(buf)
            } else {
                None
            };
        let (k_pal, k_scale): (Arc<Vec<u8>>, Arc<Vec<f32>>) = if policy.override_k_quant.is_some() {
            (
                Arc::new(identity_head_bytes.clone().unwrap()),
                Arc::new(Vec::new()),
            )
        } else {
            (
                Arc::new(k_palette_maps[job_idx].clone()),
                Arc::new(k_palette_scales[job_idx].clone()),
            )
        };
        let (v_pal, v_scale): (Arc<Vec<u8>>, Arc<Vec<f32>>) = if policy.override_v_quant.is_some() {
            (Arc::new(identity_head_bytes.unwrap()), Arc::new(Vec::new()))
        } else {
            (
                Arc::new(v_palette_maps[job_idx].clone()),
                Arc::new(v_palette_scales[job_idx].clone()),
            )
        };
        // Per-band format tags. This is the site where the chunk's formats are
        // DECIDED rather than inherited: the selection kernel picked one format
        // per (head, palette), and `alloc_side` above allocated each band's
        // destination from exactly this expression. The two must not drift —
        // the tag is what every reader will use to interpret those bytes once
        // the arena stops carrying a format (§1.5).
        let band_tag =
            |q: QuantFormat| ArenaFormatTag::from_kv_format(KvFormat::Quantized(q)).as_u8();
        let k_fmt: Arc<Vec<u8>> = Arc::new(
            (0..n_kv_head * N_PALETTE)
                .map(|i| {
                    band_tag(
                        policy
                            .override_k_quant
                            .unwrap_or(k_palette_formats[job_idx][i]),
                    )
                })
                .collect(),
        );
        let v_fmt: Arc<Vec<u8>> = Arc::new(
            (0..n_kv_head * N_PALETTE)
                .map(|i| {
                    band_tag(
                        policy
                            .override_v_quant
                            .unwrap_or(v_palette_formats[job_idx][i]),
                    )
                })
                .collect(),
        );
        // Byte size is summed from the tags just built, not from the arenas:
        // a size-class arena's slot is an upper bound on a band's bytes, never
        // the bytes themselves (`docs/archived/arena_unification.md` invariant 8).
        let byte_size = new_gids.arena_byte_size(&k_fmt, &v_fmt, elems_per_head);
        // The co-resident KV-head record is built in one batched pass after the
        // loop (below), so all quantized placements are known and the device
        // upload coalesces into a single transfer.
        full_quant_chunks.insert(
            (seq_idx, chunk_idx),
            SealedChunk {
                gids: new_gids,
                offset: src.offset,
                token_count: src.token_count,
                k_pal,
                v_pal,
                k_scale,
                v_scale,
                k_fmt,
                v_fmt,
                byte_size,
                meta: None,
            },
        );
        quant_keys.push((seq_idx, chunk_idx));
    }

    // Build the co-resident KV-head records at the quantized placement — pal/scale
    // as selected + the 8 per-palette pointers resolved against the new quant
    // arenas — in one batched, coalesced upload.
    {
        let metas = {
            let refs: Vec<ChunkRecordSrc<'_>> = quant_keys
                .iter()
                .map(|key| {
                    let c = &full_quant_chunks[key];
                    ChunkRecordSrc {
                        gids: &c.gids,
                        k_pal: c.k_pal.as_slice(),
                        v_pal: c.v_pal.as_slice(),
                        k_scale: c.k_scale.as_slice(),
                        v_scale: c.v_scale.as_slice(),
                        k_fmt: c.k_fmt.as_slice(),
                        v_fmt: c.v_fmt.as_slice(),
                    }
                })
                .collect();
            backing.build_meta_records(&refs, &arena_infos)?
        };
        for (key, meta) in quant_keys.iter().zip(metas) {
            if let Some(c) = full_quant_chunks.get_mut(key) {
                c.meta = meta;
            }
        }
    }

    // ── Merge: kernel-quantized chunks + preserve-bucket chunks ──────
    // Preserve-bucket chunks (typically cold-loaded mixed-Q chunks)
    // keep their original GIDs, arena indices, and format/pal_map/scale
    // state — no DMA, no copy. The substrate residence ends up with
    // mixed Q-format chunks per the convert kernel's output plus
    // preserve-bucket chunks per their original persist pass. Decode
    // reads them all directly via the per-(h, p) format dispatch in
    // `serialize_chunk_window_with_len`.
    let merged: Vec<SealedSequence> = sequences
        .iter()
        .enumerate()
        .map(|(seq_idx, orig)| {
            // Build partial lookup: chunk_idx → position in preserve list.
            let partial_pos_lookup: std::collections::HashMap<usize, usize> = preserve_per_seq
                [seq_idx]
                .iter()
                .enumerate()
                .map(|(pos, (chunk_idx, _))| (*chunk_idx, pos))
                .collect();
            let mut chunks: Vec<SealedChunk> = Vec::with_capacity(orig.chunks.len());
            for chunk_idx in 0..orig.chunks.len() {
                if let Some(new_chunk) = full_quant_chunks.remove(&(seq_idx, chunk_idx)) {
                    chunks.push(new_chunk);
                } else if let Some(&pos) = partial_pos_lookup.get(&chunk_idx) {
                    chunks.push(preserve_per_seq[seq_idx][pos].1.clone());
                } else {
                    return Err(candle::Error::Msg(format!(
                        "quantize_sealed_in_place: seq {seq_idx} chunk {chunk_idx} \
                         missing from both full-quant and partial-preserve buckets"
                    )));
                }
            }
            Ok(SealedSequence {
                chunks,
                token_count: orig.token_count,
                chunk_size: orig.chunk_size,
                location: ArenaLocation::Gpu,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(merged)
}

/// Inverse of [`quantize_sealed_in_place`]: decompress GPU-quant sealed chunks
/// back to GPU **F16 float** arenas, applying each `(head, palette)`'s stored
/// outer scale and un-permuting the palette map to logical dim order — i.e. the
/// exact values the attention kernel would decode, materialised into a plain
/// float arena instead of left compressed.
///
/// It drives the *same* `quantize_palette4_convert` kernel as the forward path,
/// run in reverse: the source is the sealed chunk's per-`(h, p)` quant/R16
/// sub-bands (with the chunk's stored pal_map + outer scales), the destination
/// is a fresh F16 arena with identity pal_map and unit scale. The kernel's
/// stage-1 dequant applies `src_outer` and its `xlat` gather un-permutes to
/// logical order, so the result is faithful with no host-side re-derivation.
///
/// Eligible chunks are full chunks whose every `(h, p)` GID lives in a GPU
/// `Quantized` (incl. `R16`) arena. Chunks already in GPU `Float` (partial
/// tails) pass through the preserve bucket unchanged. `head_dim` must be 128.
/// Launch the batched palette4 convert over `descs` (`[jobs × n_kv_head]`, one
/// "job" per chunk mapped onto the kernel's `num_layers` grid dim). Tiled at the
/// 65535 grid.y cap. Bit-identical to converting the jobs in separate launches —
/// only the launch grouping changes. Shared by the immediate single-layer path
/// and the deferred cross-layer path.
#[cfg(feature = "cuda")]
fn convert_descs_batched(
    head_dim: usize,
    descs: &[PalHeadDesc],
    n_kv_head: usize,
    stager: &Generation,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    if descs.is_empty() {
        return Ok(());
    }
    debug_assert_eq!(
        descs.len() % n_kv_head,
        0,
        "descs must be [jobs × n_kv_head]"
    );
    let n_jobs = descs.len() / n_kv_head;
    const MAX_GRID_Y: usize = 65_535;
    let mut base = 0usize;
    while base < n_jobs {
        let tile = (n_jobs - base).min(MAX_GRID_Y);
        let start = base * n_kv_head;
        let end = (base + tile) * n_kv_head;
        candle::quantized::cuda::quantize_palette4_convert_buffered(
            head_dim,
            &descs[start..end],
            n_kv_head,
            tile, // num_layers = job count in this tile (one chunk per block)
            1,    // num_chunks per block = 1 (each chunk is its own arena set)
            stager,
            stream,
        )?;
        base += tile;
    }
    Ok(())
}

/// Run the deferred cross-layer convert accumulated by
/// [`quantize_sealed_in_place_deferred`]: ONE batched launch over EVERY layer's
/// descriptors on the primary stream (`n_layers × n_chunks × n_kv_head` blocks),
/// filling the dst arenas the returned sequences point at, before the persist
/// pass reads them. `backing` supplies the pinned stager + CUDA device; any
/// layer's backing works (the descriptors carry their own arena pointers).
#[cfg(feature = "cuda")]
pub fn convert_deferred_descs(
    backing: &ChunkedKvBacking,
    descs: &[PalHeadDesc],
    n_kv_head: usize,
    device: &Device,
) -> Result<()> {
    if descs.is_empty() {
        return Ok(());
    }
    let cuda_dev = match device {
        Device::Cuda(d) => d,
        _ => candle::bail!("convert_deferred_descs: requires a CUDA device"),
    };
    let stager = backing.begin_stager_generation_required();
    let primary_stream = cuda_dev.cuda_stream();
    convert_descs_batched(backing.head_dim(), descs, n_kv_head, &stager, &primary_stream)?;
    drop(stager);
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn dequantize_sealed_in_place(
    backing: &ChunkedKvBacking,
    sequences: &[&SealedSequence],
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    pinned_scratch: &mut Option<PinnedBuf>,
) -> Result<Vec<SealedSequence>> {
    let _ = pinned_scratch; // kept for signature symmetry with quantize; no DtoH here.
    if sequences.is_empty() {
        return Ok(Vec::new());
    }

    let cuda_dev = match device {
        Device::Cuda(d) => d,
        _ => candle::bail!("dequantize_sealed_in_place: requires a CUDA device"),
    };

    let n_kv_head = backing.n_kv_head();
    let head_dim = backing.head_dim();
    if !candle::quantized::cuda::kvhead_supported_head_dim(head_dim) {
        candle::bail!(
            "dequantize_sealed_in_place: palette4 conversion requires head_dim 128 or 256, \
             got {head_dim}"
        );
    }
    let sub_head_dim = head_dim / N_PALETTE;
    let elems_per_head = CHUNK_SIZE * sub_head_dim;
    let pal_bytes_per_head = head_dim / 4;

    // ── Bucket each source chunk: decompress vs preserve ────────────
    // A chunk is eligible for the reverse kernel when it is full, GPU-resident,
    // and every (h, p) band is recorded as `Quantized` (R16 included — the
    // kernel's issue_load handles R16 raw). Partial tails (already GPU Float)
    // and any non-GPU chunk go to the preserve bucket unchanged. As elsewhere,
    // location comes from the arena and format from the chunk's band tags.
    let mut chunk_jobs: Vec<&SealedChunk> = Vec::new();
    let mut seq_chunk_map: Vec<(usize, usize)> = Vec::new();
    let mut preserve_per_seq: Vec<Vec<(usize, SealedChunk)>> =
        sequences.iter().map(|_| Vec::new()).collect();
    backing.inner.storage.read(|storage| {
        for (seq_idx, seq) in sequences.iter().enumerate() {
            for (chunk_idx, chunk) in seq.chunks.iter().enumerate() {
                let is_full = usize::from(chunk.token_count) >= CHUNK_SIZE;
                let mut all_gids_quant = is_full;
                if all_gids_quant {
                    for (gid, tag) in chunk.bands() {
                        // `Invalid` reports as quantized, so an unrecorded band
                        // must be rejected explicitly — the reverse kernel would
                        // otherwise be handed bytes it cannot interpret.
                        let ok = tag != ArenaFormatTag::Invalid
                            && tag.is_quantized()
                            && matches!(
                                storage.arena_key(gid.arena_idx()),
                                Some(k) if k.location == ArenaLocation::Gpu
                            );
                        if !ok {
                            all_gids_quant = false;
                            break;
                        }
                    }
                }
                if all_gids_quant {
                    seq_chunk_map.push((seq_idx, chunk_idx));
                    chunk_jobs.push(chunk);
                } else {
                    preserve_per_seq[seq_idx].push((chunk_idx, chunk.clone()));
                }
            }
        }
        Ok::<(), candle::Error>(())
    })??;
    if chunk_jobs.is_empty() {
        return Ok(sequences.iter().map(|s| (*s).clone()).collect());
    }

    let n_chunks = chunk_jobs.len();

    // ── Allocate GPU F16 destination GIDs ─────────────────────────────
    // One F16 dst GID per (chunk, head, palette, K/V) — identity layout,
    // unit scale.
    let mut new_gids_per_chunk: Vec<Vec<ChunkGid>> = Vec::with_capacity(n_chunks);
    for _ in 0..n_chunks {
        let mut chunk_gids: Vec<ChunkGid> = Vec::with_capacity(n_kv_head * GIDS_PER_HEAD);
        for _h in 0..n_kv_head {
            for _p in 0..N_PALETTE {
                let f16_key = backing
                    .inner
                    .arena_key_for(KvFormat::Float(DType::F16), ArenaLocation::Gpu)?;
                let k_gid = backing.alloc_chunk_for_key(f16_key)?;
                let v_gid = backing.alloc_chunk_for_key(f16_key)?;
                chunk_gids.push(k_gid);
                chunk_gids.push(v_gid);
            }
        }
        new_gids_per_chunk.push(chunk_gids);
    }

    // ── Build PalHeadDesc structs (src quant → dst F16) ───────────────
    let ident = identity_pal_map(head_dim);
    let mut descs: Vec<PalHeadDesc> = Vec::with_capacity(n_chunks * n_kv_head);
    backing.inner.storage.try_write(|storage| {
        for (chunk_i, src_chunk) in chunk_jobs.iter().enumerate() {
            let src_gids = &src_chunk.gids;
            for h in 0..n_kv_head {
                let mut k_src_ptrs = [0u64; N_PALETTE];
                let mut v_src_ptrs = [0u64; N_PALETTE];
                let mut k_src_fmts = [GgmlDType::F16; N_PALETTE];
                let mut v_src_fmts = [GgmlDType::F16; N_PALETTE];
                let mut k_src_scales = [1.0f32; N_PALETTE];
                let mut v_src_scales = [1.0f32; N_PALETTE];
                let mut k_dst_ptrs = [0u64; N_PALETTE];
                let mut v_dst_ptrs = [0u64; N_PALETTE];

                // Resolve one side's source pointer/format for palette `p`.
                //
                // The layout comes from the chunk's own band tag; the arena
                // supplies only the base pointer it is addressed against. The
                // tag byte round-trips through `KvFormat::from_tag`, the same
                // inverse the substrate uses, so there is no second byte→format
                // table to drift from `ArenaFormatTag::from_kv_format`.
                let resolve_src = |storage: &super::arena::ArenaStorageState,
                                   gid: &ChunkGid,
                                   tag: u8|
                 -> Result<(u64, GgmlDType)> {
                    Ok((band_ptr(storage, gid, "src")?, band_ggml_dtype(tag)?))
                };

                for p in 0..N_PALETTE {
                    let bidx = h * N_PALETTE + p;
                    let k_tag = src_chunk.k_fmt.get(bidx).copied().ok_or_else(|| {
                        candle::Error::Msg(format!("src chunk has no K format tag for band {bidx}"))
                    })?;
                    let v_tag = src_chunk.v_fmt.get(bidx).copied().ok_or_else(|| {
                        candle::Error::Msg(format!("src chunk has no V format tag for band {bidx}"))
                    })?;
                    let (kp, kf) = resolve_src(&*storage, src_gids.k_gid_pal(h, p), k_tag)?;
                    k_src_ptrs[p] = kp;
                    k_src_fmts[p] = kf;
                    let (vp, vf) = resolve_src(&*storage, src_gids.v_gid_pal(h, p), v_tag)?;
                    v_src_ptrs[p] = vp;
                    v_src_fmts[p] = vf;

                    // Stored per-(h, p) outer scale (empty scale vec ⇒ unit).
                    let sidx = bidx;
                    k_src_scales[p] = src_chunk.k_scale.get(sidx).copied().unwrap_or(1.0);
                    v_src_scales[p] = src_chunk.v_scale.get(sidx).copied().unwrap_or(1.0);

                    // F16 dst pointers.
                    let k_dst = &new_gids_per_chunk[chunk_i][h * GIDS_PER_HEAD + p * 2];
                    let v_dst = &new_gids_per_chunk[chunk_i][h * GIDS_PER_HEAD + p * 2 + 1];
                    k_dst_ptrs[p] = band_ptr(storage, k_dst, "k dst")?;
                    v_dst_ptrs[p] = band_ptr(storage, v_dst, "v dst")?;
                }

                // Source palette maps: the chunk's stored per-head slice (fall
                // back to identity when absent). Dst is identity (logical order).
                let head_pal = |pal: &[u8]| -> candle::quantized::cuda::PalMapBytes {
                    let start = h * pal_bytes_per_head;
                    let end = start + pal_bytes_per_head;
                    if pal.len() >= end {
                        let mut m = ident;
                        m[..pal_bytes_per_head].copy_from_slice(&pal[start..end]);
                        m
                    } else {
                        ident
                    }
                };

                descs.push(PalHeadDesc {
                    k_src_arena_ptrs: k_src_ptrs,
                    v_src_arena_ptrs: v_src_ptrs,
                    k_src_fmts,
                    v_src_fmts,
                    k_src_pal_map: head_pal(&src_chunk.k_pal),
                    v_src_pal_map: head_pal(&src_chunk.v_pal),
                    k_src_scales,
                    v_src_scales,
                    k_dst_arena_ptrs: k_dst_ptrs,
                    v_dst_arena_ptrs: v_dst_ptrs,
                    k_dst_fmts: [GgmlDType::F16; N_PALETTE],
                    v_dst_fmts: [GgmlDType::F16; N_PALETTE],
                    k_dst_pal_map: ident,
                    v_dst_pal_map: ident,
                    k_dst_scales: [1.0f32; N_PALETTE],
                    v_dst_scales: [1.0f32; N_PALETTE],
                });
            }
        }
        Ok(())
    })?;

    // One launch per chunk on the primary stream (same convention as the
    // forward path: per-chunk dst-state variance must not share a launch).
    let stager_generation = backing.begin_stager_generation_required();
    let primary_stream = cuda_dev.cuda_stream();
    for chunk_i in 0..n_chunks {
        let start = chunk_i * n_kv_head;
        let end = start + n_kv_head;
        candle::quantized::cuda::quantize_palette4_convert_buffered(
            head_dim,
            &descs[start..end],
            n_kv_head,
            1,
            1,
            &stager_generation,
            &primary_stream,
        )?;
    }
    let _ = copy_stream; // convert runs on primary, matching quantize_sealed_in_place.
    drop(stager_generation);

    // ── Reassemble per-sequence F16 SealedSequences ──────────────────
    // Only the dst float arenas just allocated are read below — resolve just
    // those (see the quantize path for the O(num_arenas) rationale).
    let needed_arenas: std::collections::HashSet<usize> = new_gids_per_chunk
        .iter()
        .flat_map(|gids| gids.iter().map(|g| g.arena_idx()))
        .collect();
    let arena_infos = backing.resolve_arena_info_for(&needed_arenas)?;
    // Float chunks carry an identity palette map and unit (empty) scales.
    let mut identity_head_bytes = vec![0u8; n_kv_head * pal_bytes_per_head];
    for h in 0..n_kv_head {
        identity_head_bytes[h * pal_bytes_per_head..(h + 1) * pal_bytes_per_head]
            .copy_from_slice(&ident);
    }
    let identity_head_bytes = Arc::new(identity_head_bytes);
    let empty_scale: Arc<Vec<f32>> = Arc::new(Vec::new());
    // Dequantize writes every band as F16 (see the `ArenaKey::gpu_float(F16)`
    // destination allocation above), so every band's tag is F16 — shared once
    // rather than rebuilt per chunk.
    let f16_fmt: Arc<Vec<u8>> = Arc::new(vec![
        ArenaFormatTag::from_kv_format(KvFormat::Float(
            DType::F16
        ))
        .as_u8();
        n_kv_head * N_PALETTE
    ]);

    let mut float_chunks: std::collections::HashMap<(usize, usize), SealedChunk> =
        std::collections::HashMap::with_capacity(seq_chunk_map.len());
    let mut float_keys: Vec<(usize, usize)> = Vec::with_capacity(seq_chunk_map.len());
    for (job_idx, &(seq_idx, chunk_idx)) in seq_chunk_map.iter().enumerate() {
        let new_gids = HeadGids::from_vec(new_gids_per_chunk[job_idx].clone());
        let byte_size = new_gids.arena_byte_size(&f16_fmt, &f16_fmt, elems_per_head);
        let src = chunk_jobs[job_idx];
        float_chunks.insert(
            (seq_idx, chunk_idx),
            SealedChunk {
                gids: new_gids,
                offset: src.offset,
                token_count: src.token_count,
                k_pal: identity_head_bytes.clone(),
                v_pal: identity_head_bytes.clone(),
                k_scale: empty_scale.clone(),
                v_scale: empty_scale.clone(),
                k_fmt: f16_fmt.clone(),
                v_fmt: f16_fmt.clone(),
                byte_size,
                meta: None,
            },
        );
        float_keys.push((seq_idx, chunk_idx));
    }

    // Co-resident KV-head records at the F16 placement — identity pal, unit
    // scale, F16 pointers — in one batched, coalesced upload.
    {
        let metas = {
            let refs: Vec<ChunkRecordSrc<'_>> = float_keys
                .iter()
                .map(|key| {
                    let c = &float_chunks[key];
                    ChunkRecordSrc {
                        gids: &c.gids,
                        k_pal: c.k_pal.as_slice(),
                        v_pal: c.v_pal.as_slice(),
                        k_scale: c.k_scale.as_slice(),
                        v_scale: c.v_scale.as_slice(),
                        k_fmt: c.k_fmt.as_slice(),
                        v_fmt: c.v_fmt.as_slice(),
                    }
                })
                .collect();
            backing.build_meta_records(&refs, &arena_infos)?
        };
        for (key, meta) in float_keys.iter().zip(metas) {
            if let Some(c) = float_chunks.get_mut(key) {
                c.meta = meta;
            }
        }
    }

    // Merge decompressed float chunks with preserve-bucket chunks in order.
    let merged: Vec<SealedSequence> = sequences
        .iter()
        .enumerate()
        .map(|(seq_idx, orig)| {
            let partial_pos_lookup: std::collections::HashMap<usize, usize> = preserve_per_seq
                [seq_idx]
                .iter()
                .enumerate()
                .map(|(pos, (chunk_idx, _))| (*chunk_idx, pos))
                .collect();
            let mut chunks: Vec<SealedChunk> = Vec::with_capacity(orig.chunks.len());
            for chunk_idx in 0..orig.chunks.len() {
                if let Some(new_chunk) = float_chunks.remove(&(seq_idx, chunk_idx)) {
                    chunks.push(new_chunk);
                } else if let Some(&pos) = partial_pos_lookup.get(&chunk_idx) {
                    chunks.push(preserve_per_seq[seq_idx][pos].1.clone());
                } else {
                    return Err(candle::Error::Msg(format!(
                        "dequantize_sealed_in_place: seq {seq_idx} chunk {chunk_idx} \
                         missing from both float and preserve buckets"
                    )));
                }
            }
            Ok(SealedSequence {
                chunks,
                token_count: orig.token_count,
                chunk_size: orig.chunk_size,
                location: ArenaLocation::Gpu,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(merged)
}
