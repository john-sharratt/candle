//! Quantizes a sealed sequence's eligible chunks into fresh GPU arenas and
//! reassembles the sequence in its original chunk order.
//!
//! The format of each band is decided once by the selection kernel. The
//! destination allocation and the band tags are both built from that decision,
//! so the tags a reader later uses cannot drift from the bytes written. A chunk
//! whose source is not a GPU float or R16 band goes to a preserve bucket and
//! comes back unchanged. The conversion is a compress-to-free step, so when no
//! free region exists the small quantized chunks take slots in a wider class's
//! region, and the float regions they release come back whole. A refused
//! allocation owes the whole layer, so the retry claims all of it at once.

use super::*;

pub(super) fn quantize_sealed_in_place_impl(
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
            let PolicySelection {
                k: k_candidates,
                v: v_candidates,
                thresholds: [k_threshold_hi, k_threshold_lo, v_threshold_hi, v_threshold_lo],
            } = policy_selection(policy)?;

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
    // The destination class of each band, and which bands go as one run: one
    // answer for the allocation below and for the demand a refused one owes.
    let band_keys = |fmts: &[crate::kv_cache::QuantFormat]| -> Result<Vec<ArenaKey>> {
        fmts.iter()
            .map(|f| {
                backing
                    .inner
                    .arena_key_for(KvFormat::Quantized(*f), ArenaLocation::Gpu)
            })
            .collect()
    };
    let side_formats = |chunk_i: usize, h: usize| {
        let k: Vec<_> = (0..N_PALETTE)
            .map(|p| {
                policy
                    .override_k_quant
                    .unwrap_or(k_palette_formats[chunk_i][h * N_PALETTE + p])
            })
            .collect();
        let v: Vec<_> = (0..N_PALETTE)
            .map(|p| {
                policy
                    .override_v_quant
                    .unwrap_or(v_palette_formats[chunk_i][h * N_PALETTE + p])
            })
            .collect();
        (k, v)
    };
    let allocated = (|| -> Result<Vec<Vec<ChunkGid>>> {
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
                let (k_fmts, v_fmts) = side_formats(chunk_i, h);
                // Eligibility for one contiguous run is a question about *keys*:
                // four bands in four different formats that share a size class
                // share an arena, so they can still be a run. Under per-format
                // arenas this could only pass for identical formats.
                let alloc_side = |fmts: &[crate::kv_cache::QuantFormat]| -> Result<Vec<ChunkGid>> {
                    let keys = band_keys(fmts)?;
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
        Ok(new_gids_per_chunk)
    })();
    let new_gids_per_chunk = match allocated {
        Ok(gids) => gids,
        // **Refused mid-wave: owe the whole layer, then go.** The gap between
        // forwards creates what is owed, and the pass's retry finds it all —
        // where owing only the class it was refused at made the retry stop at
        // the next arena it needed (`BackingInner::owe_capacity`). What this
        // attempt claimed went back when it failed, so the whole layer is owed.
        Err(e) if super::super::backing::BackingInner::is_wave_deferral(&e) => {
            let mut demand: Vec<(ArenaKey, usize)> = Vec::new();
            add_layer_demand(
                backing,
                policy,
                &k_palette_formats,
                &v_palette_formats,
                n_kv_head,
                &mut demand,
            )?;
            backing.inner.owe_capacity(&demand);
            return Err(e);
        }
        Err(e) => return Err(e),
    };
    if let Some(a) = alloc_ms {
        *a += t_alloc.elapsed().as_millis() as u64;
    }

    // ── Build PalHeadDesc structs + launch kernel ─────────────────────
    let mut descs: Vec<PalHeadDesc> = Vec::with_capacity(n_chunks * n_kv_head);

    backing.inner.storage.try_write(|storage| {
        let storage: &ArenaStorageState = storage;
        let mut src_addrs = BandAddrs::new(storage);
        for (chunk_i, src_bands) in chunk_jobs.iter().enumerate() {
            let ident = identity_pal_map(head_dim);
            for h in 0..n_kv_head {
                // Source — layout from the chunk's tags, addresses from the
                // arena. `bucket_quant_chunks` already proved every band is
                // Float or R16.
                let src = head_source(&mut src_addrs, src_bands, h)?;
                let mut k_dst_ptrs = [0u64; N_PALETTE];
                let mut k_dst_fmts = [GgmlDType::F16; N_PALETTE];
                let mut v_dst_ptrs = [0u64; N_PALETTE];
                let mut v_dst_fmts = [GgmlDType::F16; N_PALETTE];

                for p in 0..N_PALETTE {
                    let gid_slot = h * GIDS_PER_HEAD + p * 2;

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
                    k_src_arena_ptrs: src.k_ptrs,
                    v_src_arena_ptrs: src.v_ptrs,
                    k_src_fmts: src.k_fmts,
                    v_src_fmts: src.v_fmts,
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
                    // The same window the selection measured: the convert
                    // reads every dead slot of a partial chunk as zero.
                    valid_range: chunk_valid_ranges[chunk_i],
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
                // The source's capture, if it had one: this convert reads no Q.
                q_sign: src.q_sign.clone(),
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
