//! Decompresses quantized sealed chunks back into F16 arenas, giving the values
//! the attention kernel would decode.
//!
//! The reverse convert runs the same palette-4 kernel as the forward path. Its
//! source is each chunk's quantized sub-bands with the stored palette map and
//! outer scales, and its destination is a fresh F16 arena with an identity
//! palette map and unit scale, so the kernel un-permutes to logical dimension
//! order and no host re-derivation is needed. Only full chunks whose every band
//! is GPU-quantized are converted. Partial tails already in float pass through
//! unchanged, and a band with no recorded format is rejected, since the kernel
//! could not interpret its bytes. Converts across layers are batched into one
//! launch and give the same bytes as converting each chunk on its own.

use super::*;

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
/// tails) pass through the preserve bucket unchanged. `head_dim` must be one of
/// the compiled shapes and divide by `N_PALETTE`.
/// Launch the batched palette4 convert over `descs` (`[jobs × n_kv_head]`, one
/// "job" per chunk mapped onto the kernel's `num_layers` grid dim). Tiled at the
/// 65535 grid.y cap. Bit-identical to converting the jobs in separate launches —
/// only the launch grouping changes. Shared by the immediate single-layer path
/// and the deferred cross-layer path.
#[cfg(feature = "cuda")]
pub(super) fn convert_descs_batched(
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
    convert_descs_batched(
        backing.head_dim(),
        descs,
        n_kv_head,
        &stager,
        &primary_stream,
    )?;
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
                let resolve_src = |storage: &ArenaStorageState,
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
                    // The bucket above admits full chunks only.
                    valid_range: FULL_CHUNK_WINDOW,
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
                // F16 bands hold no Q either: the source's capture stands.
                q_sign: src.q_sign.clone(),
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
