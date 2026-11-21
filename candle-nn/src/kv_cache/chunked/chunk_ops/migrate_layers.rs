//! Builds warm copies of cold chunks, and moves a layer's hot chunks to host
//! memory in staging groups that fit the span.
//!
//! Cold chunks are rebuilt in host memory with their format tags, palette maps and
//! scales as they were written. The hot-to-warm copy stages every layer's bands in
//! groups bounded by the span, so no single gather exceeds the reservation, and
//! each group is copied with one device-to-host transfer. Scratch grows only after
//! the span has granted the group, since pinned memory never shrinks.

use super::*;

/// One sealed chunk as the log holds it, borrowed from a read buffer: what
/// [`ChunkedKvBacking::warm_from_cold`] rebuilds a warm copy from.
#[derive(Debug, Clone, Copy)]
pub struct ColdChunk<'a> {
    /// The window's skip-count from the start of the chunk.
    pub offset: u16,
    /// Valid tokens in the window.
    pub token_count: u16,
    /// Per-`(head, palette)` K and V format tags, `n_kv_head × N_PALETTE` each.
    pub k_fmt: &'a [u8],
    pub v_fmt: &'a [u8],
    /// Packed palette maps and outer scales, as the chunk recorded them.
    pub k_pal: &'a [u8],
    pub v_pal: &'a [u8],
    pub k_scale: &'a [f32],
    pub v_scale: &'a [f32],
    /// Every band's payload, in `HeadGids` order.
    pub kv_bytes: &'a [u8],
}

impl ChunkedKvBacking {
    /// Build a warm (host RAM) copy of one layer's sealed chunks from their
    /// cold images, without touching the device.
    ///
    /// Each chunk is laid out as the log holds it (`ChunkPayload`): its
    /// per-`(head, palette)` K and V format tags, palette maps and scales, and
    /// one blob holding every band's payload in `HeadGids` order —
    /// `(h * N_PALETTE + p) * 2 + is_value` — each as long as its own tag says.
    /// Every band takes a slot of its own in the warm tier, as a cold load
    /// gives it one on the device, and the chunk carries no device record.
    #[cfg(feature = "cuda")]
    pub fn warm_from_cold(&self, chunks: &[ColdChunk<'_>]) -> candle::Result<SealedSequence> {
        let n_kv_head = self.inner.n_kv_head;
        let want = n_kv_head * N_PALETTE;
        let elems = self.inner.elems_per_chunk();

        // Every band's key and byte range in its chunk's blob, checked before
        // anything is allocated, so a malformed image leaves the tier untouched.
        let mut plans: Vec<Vec<(ArenaKey, std::ops::Range<usize>)>> =
            Vec::with_capacity(chunks.len());
        for (c, chunk) in chunks.iter().enumerate() {
            if chunk.k_fmt.len() != want || chunk.v_fmt.len() != want {
                candle::bail!(
                    "warm_from_cold: chunk {c} has {} K and {} V band formats, expected {want}",
                    chunk.k_fmt.len(),
                    chunk.v_fmt.len()
                );
            }
            let mut bands = Vec::with_capacity(want * 2);
            let mut off = 0usize;
            for sub in 0..want {
                for tag in [chunk.k_fmt[sub], chunk.v_fmt[sub]] {
                    let fmt = KvFormat::from_tag(tag).ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "warm_from_cold: chunk {c} band tag {tag} names no storage format"
                        ))
                    })?;
                    let len = payload_bytes_for_tag(ArenaFormatTag::from_u8(tag), elems)
                        .ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "warm_from_cold: chunk {c} band tag {tag} has no payload length"
                            ))
                        })?;
                    let key = self.inner.arena_key_for(fmt, ArenaLocation::Cpu)?;
                    bands.push((key, off..off + len));
                    off += len;
                }
            }
            if off != chunk.kv_bytes.len() {
                candle::bail!(
                    "warm_from_cold: chunk {c}'s bands need {off} bytes, its image holds {}",
                    chunk.kv_bytes.len()
                );
            }
            plans.push(bands);
        }

        // One slot per band (one state.write), then every band's bytes (one
        // storage.write), as the hot→warm copy does.
        let gids: Vec<Vec<ChunkGid>> = {
            let _state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            plans
                .iter()
                .map(|bands| {
                    bands
                        .iter()
                        .map(|(key, _)| self.alloc_chunk_for_key(*key))
                        .collect::<candle::Result<Vec<ChunkGid>>>()
                })
                .collect::<candle::Result<_>>()?
        };
        self.inner.storage.try_write(|s| -> candle::Result<()> {
            for ((chunk, bands), chunk_gids) in chunks.iter().zip(&plans).zip(&gids) {
                for ((_, range), gid) in bands.iter().zip(chunk_gids) {
                    Self::write_chunk_from_pinned_bytes(
                        s.arenas_mut(),
                        gid.arena_idx(),
                        gid.chunk_idx(),
                        &chunk.kv_bytes[range.clone()],
                    )?;
                }
            }
            Ok(())
        })?;

        let sealed: Vec<SealedChunk> = chunks
            .iter()
            .zip(gids)
            .map(|(chunk, chunk_gids)| SealedChunk {
                gids: HeadGids::from_vec(chunk_gids),
                offset: chunk.offset,
                token_count: chunk.token_count,
                k_pal: Arc::new(chunk.k_pal.to_vec()),
                v_pal: Arc::new(chunk.v_pal.to_vec()),
                k_scale: Arc::new(chunk.k_scale.to_vec()),
                v_scale: Arc::new(chunk.v_scale.to_vec()),
                k_fmt: Arc::new(chunk.k_fmt.to_vec()),
                v_fmt: Arc::new(chunk.v_fmt.to_vec()),
                byte_size: chunk.kv_bytes.len() as u64,
                meta: None,
                // Read back from the log: its signature was folded at its seal.
                q_sign: None,
            })
            .collect();
        Ok(SealedSequence {
            token_count: sealed.iter().map(|c| c.token_count as usize).sum(),
            chunks: sealed,
            chunk_size: CHUNK_SIZE,
            location: ArenaLocation::Cpu,
        })
    }

    /// Cross-layer batched hot→warm DtoH. The per-layer
    /// [`Self::migrate_sealed_to_cpu_batch_async`] issues a gather + DtoH + sync
    /// **per backing** (48× per persist pass on Qwen3); this resolves EVERY
    /// layer's bands in one walk and stages them in `ceil(total / span)` groups
    /// ([`cross_layer_staging_groups`]) — one gather + one DtoH + one
    /// `copy_stream.synchronize()` per group — scattering each group's bands into
    /// fresh CPU arenas layer by layer. Collapses the WDDM per-launch/per-sync
    /// overhead that made `copy_ms` the hot→warm drain bottleneck, and fits the
    /// staging span whatever one layer holds. `backings[i]` owns
    /// `per_layer_seqs[i]`; returns the warm (CPU) sequences per layer, in the
    /// same order. Bit-identical to the per-layer path — proven by
    /// `migrate_layers_deferred_matches_immediate_bytes` extended, and the
    /// dedicated `migrate_layers_batched_matches_per_layer_bytes`.
    #[cfg(feature = "cuda")]
    pub fn migrate_sealed_layers_to_cpu_batch(
        backings: &[&ChunkedKvBacking],
        device: &Device,
        copy_stream: &Arc<CudaStream>,
        pinned_scratch: &mut Option<PinnedBuf>,
        per_layer_seqs: &[Vec<&SealedSequence>],
    ) -> candle::Result<Vec<Vec<SealedSequence>>> {
        Self::migrate_sealed_layers_to_cpu_batch_in(
            backings,
            device,
            copy_stream,
            pinned_scratch,
            per_layer_seqs,
            MIGRATION_STAGING_CAP_BYTES,
        )
    }

    /// [`Self::migrate_sealed_layers_to_cpu_batch`] staging in groups of at most
    /// `cap` bytes — the span's size in production; a test passes a few bands'
    /// worth so its small sequences split mid-layer exactly as a 128K turn's
    /// layer splits against the span.
    #[cfg(feature = "cuda")]
    pub(crate) fn migrate_sealed_layers_to_cpu_batch_in(
        backings: &[&ChunkedKvBacking],
        device: &Device,
        copy_stream: &Arc<CudaStream>,
        pinned_scratch: &mut Option<PinnedBuf>,
        per_layer_seqs: &[Vec<&SealedSequence>],
        cap: usize,
    ) -> candle::Result<Vec<Vec<SealedSequence>>> {
        use candle::cuda_backend::WrapErr;
        use std::collections::{HashMap, HashSet};

        use super::super::migrate::{kv_migrate_on, MigrationPlan};

        if backings.len() != per_layer_seqs.len() {
            return Err(candle::Error::Msg(
                "migrate_sealed_layers_to_cpu_batch: backings/seqs length mismatch".into(),
            ));
        }
        match device {
            candle::Device::Cuda(_) => {}
            _ => {
                return Err(candle::Error::Msg(
                    "migrate_sealed_layers_to_cpu_batch requires a CUDA device".into(),
                ))
            }
        }

        // Per-layer resolution kept for the scatter phase. Offsets are assigned
        // per staging group below (not globally), so the transient GPU staging
        // buffer never scales with the whole cross-layer batch.
        struct LayerResolve {
            unique_raws: Vec<i64>,
            /// Source device ptr + byte length per unique raw gid.
            src: HashMap<i64, (i64, usize)>,
            src_keys: HashMap<i64, ArenaKey>,
        }

        // ── Resolve every layer's sources (device ptr + len per gid) ────
        let arena_chunks = GID_STRIDE;
        let mut layers: Vec<LayerResolve> = Vec::with_capacity(backings.len());
        let mut grand_total: usize = 0;
        for (li, backing) in backings.iter().enumerate() {
            let arena_info = backing.resolve_arena_info()?;
            let mut seen: HashSet<i64> = HashSet::new();
            let mut unique_raws: Vec<i64> = Vec::new();
            let mut src: HashMap<i64, (i64, usize)> = HashMap::new();
            let mut src_keys: HashMap<i64, ArenaKey> = HashMap::new();
            let elems = backing.inner.elems_per_chunk();
            for seq in &per_layer_seqs[li] {
                for chunk in &seq.chunks {
                    for (gid, tag) in chunk.bands() {
                        let raw = gid.raw();
                        if !seen.insert(raw) {
                            continue;
                        }
                        let arena_idx = (raw as usize) / arena_chunks;
                        let chunk_idx = (raw as usize) % arena_chunks;
                        let info = arena_info.get(arena_idx).ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "migrate_sealed_layers_to_cpu_batch: layer {li} arena {arena_idx} out of range"
                            ))
                        })?;
                        if info.base_ptr == 0 {
                            return Err(candle::Error::Msg(
                                "migrate_sealed_layers_to_cpu_batch: source arena not GPU-resident"
                                    .into(),
                            ));
                        }
                        // Payload from the band's tag, stride from the arena.
                        let len = payload_bytes_for_tag(tag, elems).ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "migrate_sealed_layers_to_cpu_batch: band tag {tag:?} names no                                  storage format, so its byte length is unknown"
                            ))
                        })?;
                        let ptr = info.base_ptr as i64 + chunk_idx as i64 * info.chunk_byte_stride;
                        src.insert(raw, (ptr, len));
                        unique_raws.push(raw);
                        grand_total += len;
                        let key = backing
                            .inner
                            .storage
                            .read(|s| s.arena_key(arena_idx))?
                            .ok_or_else(|| {
                                candle::Error::Msg(format!(
                                    "migrate_sealed_layers_to_cpu_batch: layer {li} arena key {arena_idx} not found"
                                ))
                            })?;
                        src_keys.insert(raw, key);
                    }
                }
            }
            layers.push(LayerResolve {
                unique_raws,
                src,
                src_keys,
            });
        }

        if grand_total == 0 {
            return Ok(per_layer_seqs
                .iter()
                .map(|seqs| seqs.iter().map(|s| (*s).clone()).collect())
                .collect());
        }

        // ── Stage in band groups that each fit the span, across layers ──
        // A single cross-layer gather would size the transient GPU staging to the
        // ENTIRE batch — under ingest pressure that alloc can OOM the very copy
        // meant to relieve pressure — and grouping whole layers left a lone layer
        // wider than the span (one layer of a 128K turn is ~84 MB at C5) with
        // nowhere smaller to go. So the bands of every layer, in layer order, are
        // grouped to the span: ONE gather + DtoH + sync per group, `ceil(total /
        // cap)` of them, every group fitting by construction.
        let layer_lens: Vec<Vec<usize>> = layers
            .iter()
            .map(|l| l.unique_raws.iter().map(|r| l.src[r].1).collect())
            .collect();
        let groups = cross_layer_staging_groups(&layer_lens, cap);
        let mut new_gids: Vec<HashMap<i64, ChunkGid>> =
            (0..backings.len()).map(|_| HashMap::new()).collect();
        let domain = bump_arena::persistence_domain(copy_stream)?;
        // The copy's two halves, timed apart: the device side (gather, DtoH and
        // the sync) and the host side (dest gids and the scatter into CPU arenas).
        let mut transfer = std::time::Duration::ZERO;
        let mut scatter = std::time::Duration::ZERO;
        let mut bands = 0usize;
        for group in &groups {
            let t_transfer = std::time::Instant::now();
            // One generation per group: dropping it fences the copy stream and
            // rewinds the cursor, so the next group reuses the same bytes. The
            // range below borrows it, so it cannot outlive the generation.
            let staging_gen = domain.generation(NOT_A_WAVE, NOT_A_WAVE)?;
            let staging = staging_gen.alloc(group.bytes, 256)?;
            let staging_base = staging.ptr as i64;
            // **Host scratch only once the span has granted the group.** Pinned
            // memory grows and never shrinks, so growing it ahead of a staging the
            // span then refuses — a lone band wider than the cap is grouped alone
            // — would leave an oversized buffer held for the process, and every
            // later pass would repeat the refusal over it. The persistence thread
            // preallocates the span's size, so a group that fits the span needs no
            // growth here at all.
            if pinned_scratch
                .as_ref()
                .is_none_or(|b| b.len() < group.bytes)
            {
                *pinned_scratch = Some(PinnedBuf::alloc_default_or_host_fallible(group.bytes)?);
            }
            let mut plan = MigrationPlan::new();
            let mut off = 0i64;
            for run in &group.runs {
                let layer = &layers[run.layer];
                for raw in &layer.unique_raws[run.start..run.end] {
                    let (ptr, len) = layer.src[raw];
                    plan.push(ptr, staging_base + off, len as i64);
                    off += len as i64;
                }
            }
            kv_migrate_on(device, &plan, Some(copy_stream))?;
            {
                let scratch = pinned_scratch
                    .as_mut()
                    .expect("pinned scratch allocated above");
                let dst = &mut scratch.as_mut_slice()[..group.bytes];
                // SAFETY: as the hot->warm site — a range of this domain's span
                // held by `staging_gen`, which `staging` borrows, so it is
                // still open here by construction; copied into pinned host
                // scratch, synced here.
                unsafe {
                    candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async(
                        dst,
                        staging_base as u64,
                        copy_stream.cu_stream(),
                    )
                }
                .map_err(|e| candle::Error::Msg(format!("layer staging readback: {e}")))?;
                copy_stream.synchronize().w()?;
            }
            drop(staging_gen);
            transfer += t_transfer.elapsed();
            let t_scatter = std::time::Instant::now();

            // ── Scatter this group's bands into fresh CPU arenas ───────────
            let scratch_ref = pinned_scratch
                .as_ref()
                .expect("pinned scratch allocated above");
            let mut run_off = 0usize;
            for run in &group.runs {
                let backing = backings[run.layer];
                let resolve = &layers[run.layer];
                let raws = &resolve.unique_raws[run.start..run.end];
                let gids = &mut new_gids[run.layer];
                // Allocate dest CPU GIDs (one state.write per run), one bulk claim
                // per destination format rather than one claim per band: the per-band
                // claim was most of the scatter's ~1.4 µs a band.
                {
                    let _state = backing
                        .state
                        .write()
                        .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
                    let mut by_key: HashMap<ArenaKey, Vec<i64>> = HashMap::new();
                    for &raw in raws {
                        by_key
                            .entry(resolve.src_keys[&raw].at(ArenaLocation::Cpu))
                            .or_default()
                            .push(raw);
                    }
                    for (cpu_key, keyed) in by_key {
                        let claimed = backing.alloc_chunks_for_key_bulk(cpu_key, keyed.len())?;
                        gids.extend(keyed.into_iter().zip(claimed));
                    }
                }
                // Write this run's slice of the group blob → CPU arena.
                let base = run_off;
                backing.inner.storage.try_write(|s| -> candle::Result<()> {
                    let mut at = base;
                    for &raw in raws {
                        let len = resolve.src[&raw].1;
                        let bytes = &scratch_ref.as_slice()[at..at + len];
                        at += len;
                        let new_gid = &gids[&raw];
                        Self::write_chunk_from_pinned_bytes(
                            s.arenas_mut(),
                            new_gid.arena_idx(),
                            new_gid.chunk_idx(),
                            bytes,
                        )?;
                    }
                    Ok(())
                })?;
                run_off += raws.iter().map(|r| resolve.src[r].1).sum::<usize>();
                bands += raws.len();
            }
            scatter += t_scatter.elapsed();
        }
        tracing::debug!(
            target: "candle_nn::kv_cache::migrate",
            groups = groups.len(),
            bands,
            mib = grand_total >> 20,
            transfer_ms = transfer.as_millis() as u64,
            scatter_ms = scatter.as_millis() as u64,
            "hot→warm copy: device transfer and host scatter"
        );

        // ── Rebuild each layer's SealedSequences with its mapped CPU GIDs ──
        // A layer with no bands carried no GPU bytes and is handed back as it
        // came, exactly as the per-layer path hands back a layer with nothing to
        // copy.
        let mut out: Vec<Vec<SealedSequence>> = Vec::with_capacity(backings.len());
        for (li, gids) in new_gids.iter().enumerate() {
            if layers[li].unique_raws.is_empty() {
                out.push(per_layer_seqs[li].iter().map(|s| (*s).clone()).collect());
                continue;
            }
            let seqs: candle::Result<Vec<SealedSequence>> = per_layer_seqs[li]
                .iter()
                .map(|seq| -> candle::Result<SealedSequence> {
                    let new_chunks: candle::Result<Vec<SealedChunk>> = seq
                        .chunks
                        .iter()
                        .map(|chunk| {
                            let mapped =
                                chunk.gids.map_unique(|gid| Ok(gids[&gid.raw()].clone()))?;
                            Ok(SealedChunk {
                                gids: mapped,
                                meta: None,
                                ..chunk.clone()
                            })
                        })
                        .collect();
                    Ok(SealedSequence {
                        chunks: new_chunks?,
                        token_count: seq.token_count,
                        chunk_size: seq.chunk_size,
                        location: ArenaLocation::Cpu,
                    })
                })
                .collect();
            out.push(seqs?);
        }
        Ok(out)
    }
}
