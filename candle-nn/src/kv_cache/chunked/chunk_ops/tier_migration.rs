//! Sets the batch size for moving chunks between tiers, and the staging groups
//! that realise it.
//!
//! Staging is batched against a fixed cap rather than one allocation, so a pass
//! wider than the span is split into groups, and each group opens its own
//! generation. The cap is charged to the reservation, so it is sized from the
//! smallest group batching can reduce to, not from a measured peak. The
//! quantise-on-evict path ingests only float or R16 sources directly, and any
//! other format takes the format-preserving migration.

use super::*;

// Shared production thresholds are defined in compression_policy.rs and are
// consumed here directly so the runtime and table harness stay in sync.

/// Upper bound on one hot→warm migration batch's contiguous staging buffer
/// (host pinned/heap AND transient GPU staging). Kept deliberately small: the
/// warm tier fills host RAM with thousands of 16 MiB CPU arenas, so a *large*
/// contiguous host allocation (the old 2 GiB) can fail on a fragmented heap and
/// abort the process. The persistence thread pre-sizes its reusable scratch to
/// this at startup (clean heap), so it never has to re-grow under fragmentation.
///
/// **This is charged to the reservation, so it is priced in KV regions** — the
/// GPU half of the staging comes out of the persistence bump domain, whose span
/// is exactly this cap ([`super::bump_arena::persistence_domain`]), and the
/// transient tier is subtracted from the KV side before a single region is
/// carved. 512 MiB costs 32 regions of KV, which on a 16 GiB card is real, and
/// the measured GPU staging peak on a decode/ingest workload is **29,696 B**.
///
/// **Both staging sites batch against it**, so it is a batch size rather than a
/// ceiling and the span bounds one group instead of a whole pass:
/// `migrate_sealed_layers_to_cpu_batch` groups by layer, and the hot→warm gather
/// groups by band through [`staging_groups`]. Each group opens its own
/// generation, so the cursor rewinds between them and peak transient use is one
/// group. The warm→hot elevate stages nothing on the device: its scatter reads
/// the pinned ring in place (`docs/warm_lift.md`).
///
/// 64 MiB is sized from the floor the batching can shrink to — a single ~30 MB
/// layer, so two layers plus headroom — not from the 29,696 B observed peak,
/// which was measured on runs where the elevate path never executed
/// (`residences=0`, persist domain `cursor=0 peak=0`) and is therefore a
/// measurement of this path *not running*. The cost is DtoH syncs on the
/// hot→warm path (~22 for a ~1.4 GiB pass against the per-layer 48 that made
/// `copy_ms` the drain bottleneck), paid off the decode path; the gain is
/// 28 KV regions that the expert cache and context depth get instead.
pub const MIGRATION_STAGING_CAP_BYTES: usize = 64 * 1024 * 1024;

/// Split consecutive per-band byte lengths into groups that each fit `cap`.
///
/// Returns `(start, end, bytes)` per group, `end` exclusive. **Always takes at
/// least one element**, so a single band wider than `cap` forms its own
/// oversized group and makes progress instead of looping forever — the same
/// rule `migrate_sealed_layers_to_cpu_batch` applies to layers.
///
/// Shared by both staging sites so the two cannot drift, and pure so the
/// boundary arithmetic is testable without a GPU
/// ([`staging_groups_respect_the_cap`](tests)).
// Without `cuda` the migration paths that call this are compiled out, so the
// test above is its only consumer — which is the point of keeping it pure.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(in super::super) fn staging_groups(lens: &[usize], cap: usize) -> Vec<(usize, usize, usize)> {
    let mut groups = Vec::new();
    let mut i = 0usize;
    while i < lens.len() {
        let mut j = i;
        let mut bytes = 0usize;
        while j < lens.len() {
            if j > i && bytes + lens[j] > cap {
                break;
            }
            bytes += lens[j];
            j += 1;
        }
        groups.push((i, j, bytes));
        i = j;
    }
    groups
}

/// A run of consecutive bands of one layer inside a cross-layer staging group:
/// `layer`'s bands `start..end`, in that layer's band order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) struct LayerRun {
    pub layer: usize,
    pub start: usize,
    pub end: usize,
}

/// One gather + DtoH of a cross-layer migrate: its bytes, and the layer runs it
/// carries in staging order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct StagingGroup {
    pub bytes: usize,
    pub runs: Vec<LayerRun>,
}

/// Split every layer's band lengths, taken in layer order, into groups that each
/// fit `cap` — [`staging_groups`] over the bands of all layers at once.
///
/// **By band, not by layer.** A layer's hot set is not bounded by the staging
/// span — one layer of a 128K turn is ~84 MB at C5 against the 64 MiB span — so
/// grouping whole layers left the floor at one oversized layer, which the span
/// refused. A band is one chunk's palette payload, kilobytes, so every group fits
/// by construction, and the group count stays `ceil(total / cap)`.
///
/// A layer with no bands contributes no run. Pure, so the boundary arithmetic is
/// testable without a GPU ([`cross_layer_groups_split_a_layer_wider_than_the_cap`](tests)).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(in super::super) fn cross_layer_staging_groups(
    layer_lens: &[Vec<usize>],
    cap: usize,
) -> Vec<StagingGroup> {
    let at: Vec<(usize, usize)> = layer_lens
        .iter()
        .enumerate()
        .flat_map(|(layer, lens)| (0..lens.len()).map(move |band| (layer, band)))
        .collect();
    let lens: Vec<usize> = layer_lens.iter().flatten().copied().collect();
    staging_groups(&lens, cap)
        .into_iter()
        .map(|(i, j, bytes)| {
            let mut runs: Vec<LayerRun> = Vec::new();
            for &(layer, band) in &at[i..j] {
                match runs.last_mut() {
                    Some(run) if run.layer == layer => run.end = band + 1,
                    _ => runs.push(LayerRun {
                        layer,
                        start: band,
                        end: band + 1,
                    }),
                }
            }
            StagingGroup { bytes, runs }
        })
        .collect()
}

/// Source formats the quantize-on-evict kernel can ingest directly: `Float`
/// (any dtype) or `Quantized(R16)`. Anything else has to take the
/// format-preserving migration path because the selection / conversion kernels
/// are only compiled for these source layouts.
///
/// Takes the band's own format tag, read from the chunk rather than from the
/// arena its gid points into — an arena holds whatever fits its size class and
/// cannot answer the question (`docs/archived/arena_unification.md` principle 8).
/// Callers pair this with the arena's `location`, which *is* arena identity.
#[cfg(feature = "cuda")]
#[inline]
pub(in super::super) fn needs_reconcile_source_tag(tag: ArenaFormatTag) -> bool {
    // Every float tag qualifies; of the quantized tags only R16. `Invalid`
    // reports as quantized, so it correctly fails both arms.
    !tag.is_quantized() || tag == ArenaFormatTag::R16
}

impl ChunkedKvBacking {
    // ── Tiered-storage sealed-sequence migration ──────────────────────────────

    /// Migrate one sealed sequence from the GPU (hot) tier to the CPU (warm) tier.
    ///
    /// Each unique [`ChunkGid`] in the sequence's [`HeadGids`] is migrated exactly
    /// once to the CPU arena that matches the source format (same `KvFormat`,
    /// `ArenaLocation::Cpu`).  Slots that share a raw id (e.g. all K-palette slots
    /// in the float case) receive a clone of the same CPU GID — sharing structure
    /// is preserved by [`HeadGids::map_unique`].
    ///
    /// Returns a new `SealedSequence` whose chunks reference CPU-resident arenas.
    /// The original GPU chunks are kept alive by the caller's `Arc` until it drops
    /// them; RAII then returns them to the GPU pool automatically.
    ///
    /// # Async note
    /// The underlying `to_device` call issues a CUDA DMA transfer on the current
    /// stream.  The CPU does not block — the copy runs on the GPU timeline and
    /// completes before any subsequent GPU operation on that stream reads the data.
    /// Future work: route through a dedicated copy stream + pinned host buffers
    /// so GPU compute and PCIe transfer overlap fully.
    pub fn migrate_sealed_to_cpu(&self, sealed: &SealedSequence) -> candle::Result<SealedSequence> {
        let cpu_chunks: candle::Result<Vec<SealedChunk>> = sealed
            .chunks
            .iter()
            .map(|chunk| {
                let new_gids = chunk.gids.map_unique(|gid| {
                    let cpu_key = self.inner.storage.read(|s| {
                        s.arena_key(gid.arena_idx())
                            .map(|k| k.at(ArenaLocation::Cpu))
                            .ok_or_else(|| {
                                candle::Error::Msg(format!(
                                    "migrate_sealed_to_cpu: arena {} not found",
                                    gid.arena_idx()
                                ))
                            })
                    })??;
                    self.migrate_chunk(gid.raw(), cpu_key)
                })?;
                Ok(SealedChunk {
                    gids: new_gids,
                    // Demote (GPU→CPU): the CPU/warm tier has no device-resident
                    // record, so drop the handle. The record is rebuilt at the
                    // next elevate; while warm the chunk uses scratch heads.
                    meta: None,
                    ..chunk.clone()
                })
            })
            .collect();
        Ok(SealedSequence {
            chunks: cpu_chunks?,
            token_count: sealed.token_count,
            chunk_size: sealed.chunk_size,
            location: ArenaLocation::Cpu,
        })
    }

    /// Batched VRAM→RAM migration for many sealed sequences at once.
    ///
    /// Functionally equivalent to calling [`Self::migrate_sealed_to_cpu`]
    /// once per sequence, but amortises the heavy locking:
    ///
    /// - The per-layer `state.write()` is acquired **once** for the
    ///   whole batch instead of once per migrated chunk.
    /// - The cross-layer `inner.storage.try_write()` is acquired **once**
    ///   for the whole batch instead of once per migrated chunk.
    ///
    /// For a hot→warm pass over N residences × C chunks × G GIDs-per-chunk,
    /// this drops `O(N·C·G)` lock acquisitions to `O(1)` on the data-copy
    /// path — the dominant cost on small chunks where the DMA itself is
    /// already pipelined on the default stream.
    ///
    /// The per-chunk data copy still uses the existing
    /// [`Self::copy_chunk_data_static`] / [`Self::convert_chunk_data_static`]
    /// path under that single storage write lock. The bytes flow through
    /// `Tensor::copy` (DtoH on the current stream) per chunk; the driver
    /// pipelines them, so they overlap naturally. Future revisions can
    /// route the gather through a single [`super::migrate::kv_migrate_on`]
    /// call on a dedicated copy stream once a bytes-into-CPU-arena ingest
    /// API exists.
    pub fn migrate_sealed_to_cpu_batch(
        &self,
        sequences: &[&SealedSequence],
    ) -> candle::Result<Vec<SealedSequence>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }

        // Walk every chunk in every sequence, collecting unique source
        // GIDs. `HeadGids` already dedups within one chunk via
        // `map_unique`, but the same GID can appear in multiple chunks
        // (and the same chunk can appear in multiple sequences when the
        // caller is migrating overlapping work — unusual but legal).
        let mut unique_raws: Vec<i64> = Vec::new();
        let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for seq in sequences {
            for chunk in &seq.chunks {
                for gid in chunk.gids.0.iter() {
                    if seen.insert(gid.raw()) {
                        unique_raws.push(gid.raw());
                    }
                }
            }
        }

        let arena_chunks = GID_STRIDE;

        // Phase 1: one `state.write()` for the batch — resolve every
        // source's arena key (the CPU dest just swaps location) and
        // allocate every destination GID up front.
        let mut new_gids: std::collections::HashMap<i64, ChunkGid> =
            std::collections::HashMap::new();
        let mut src_keys: std::collections::HashMap<i64, ArenaKey> =
            std::collections::HashMap::new();
        {
            let _state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for &raw in &unique_raws {
                let arena_idx = (raw as usize) / arena_chunks;
                let key = self
                    .inner
                    .storage
                    .read(|s| s.arena_key(arena_idx))?
                    .ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "migrate_sealed_to_cpu_batch: arena {arena_idx} not found"
                        ))
                    })?;
                let cpu_key = key.at(ArenaLocation::Cpu);
                src_keys.insert(raw, key);
                new_gids.insert(raw, self.alloc_chunk_for_key(cpu_key)?);
            }
        }

        // Phase 2: one `storage.try_write()` for the batch — every
        // per-chunk data copy runs back-to-back with no relock.
        self.inner.storage.try_write(|s| -> candle::Result<()> {
            for &raw in &unique_raws {
                let src_arena_idx = (raw as usize) / arena_chunks;
                let src_chunk_idx = (raw as usize) % arena_chunks;
                // Hole-aware existence check (map lookup), NOT index-vs-len:
                // `arena_count()` is the map LEN, which undercounts once any
                // arena has been tombstoned — the old bound misjudged valid
                // high-index sources as "tombstoned" and SKIPPED their copy,
                // leaving the freshly-allocated destination chunk with
                // UNINITIALIZED contents that were then stored as warm KV and
                // later attended over (silent garbage-context corruption). The
                // batch holds every source chunk's gid Arc, so a genuinely
                // absent arena is an arena freed with live KV — fail loudly.
                if s.arena_key(src_arena_idx).is_none() {
                    return Err(candle::Error::Msg(format!(
                        "hot→warm batch: source arena {src_arena_idx} (gid {raw}) vanished \
                         while its chunk is still referenced — arena freed with live KV"
                    )));
                }
                let src_key = src_keys[&raw];
                let cpu_key = src_key.at(ArenaLocation::Cpu);
                let new_gid = new_gids[&raw].clone();
                // Same class either side — the destination key only swaps the
                // location — so this is a byte-verbatim slot relocation.
                debug_assert_eq!(src_key.class, cpu_key.class);
                Self::copy_slot_bytes(
                    s.arenas_mut(),
                    src_arena_idx,
                    src_chunk_idx,
                    new_gid.arena_idx(),
                    new_gid.chunk_idx(),
                )?;
            }
            Ok(())
        })?;

        // Phase 3: rebuild each `SealedSequence` with mapped GIDs. The
        // shape (per-chunk metadata, head-gid topology) is preserved
        // verbatim; only the GIDs themselves change.
        sequences
            .iter()
            .map(|seq| -> candle::Result<SealedSequence> {
                let new_chunks: candle::Result<Vec<SealedChunk>> = seq
                    .chunks
                    .iter()
                    .map(|chunk| {
                        let mapped = chunk
                            .gids
                            .map_unique(|gid| Ok(new_gids[&gid.raw()].clone()))?;
                        Ok(SealedChunk {
                            gids: mapped,
                            // Demote (GPU→CPU): no device-resident record on the
                            // CPU tier; the chunk uses scratch heads while warm.
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
            .collect()
    }

    /// **Fully-batched** VRAM→RAM migration for many sealed sequences on
    /// a dedicated CUDA copy stream with a pinned host staging buffer.
    ///
    /// This is the production hot→warm path. It replaces the per-chunk
    /// [`Tensor::copy`] / `slice_set` pipeline (which serialises N
    /// individual `cudaMemcpyDtoH` calls on the default stream) with a
    /// single batched device-side gather followed by a single
    /// `cudaMemcpyDtoHAsync` on `copy_stream`:
    ///
    /// ```text
    ///   GPU arena chunks ──kv_migrate_on──▶ device staging ──memcpy_dtoh──▶ pinned host scratch
    ///                       (copy stream)                    (copy stream)
    ///                          ┃                                 ┃
    ///                          └───────  one stream sync ────────┘
    ///                                                            │
    ///                                                            ▼
    ///                                       Tensor::from_slice + slice_set
    ///                                       (CPU-only, under one lock)
    /// ```
    ///
    /// Compared with [`Self::migrate_sealed_to_cpu`] called per
    /// sequence:
    ///
    /// - **One** kernel launch (gather) and **one** `cudaMemcpy`
    ///   (DtoH) total, instead of N per sequence.
    /// - **Dedicated copy stream**: GPU compute on the main stream
    ///   overlaps with this thread's DMA. The driver's copy engine
    ///   pipelines transfers without blocking decode.
    /// - **Pinned host staging**: write-combined `cuMemHostAlloc`'d
    ///   memory, no pageable bounce buffer inside the driver.
    /// - **One** state-lock + **one** storage-lock acquisition for
    ///   the whole batch.
    ///
    /// `pinned_scratch` is grown on demand: pass an `&mut Option<…>`
    /// that the persistence thread owns and re-uses across passes.
    ///
    /// **Format support:** Float arenas (BF16/F16/F32) only on this
    /// fast path. Quantized arenas have no raw-bytes ingest API on
    /// CPU `QTensor`; callers needing them must fall back to
    /// [`Self::migrate_sealed_to_cpu_batch`] (lock-batched, on the
    /// default stream).
    #[cfg(feature = "cuda")]
    pub fn migrate_sealed_to_cpu_batch_async(
        &self,
        device: &Device,
        copy_stream: &Arc<CudaStream>,
        pinned_scratch: &mut Option<PinnedBuf>,
        sequences: &[&SealedSequence],
    ) -> candle::Result<Vec<SealedSequence>> {
        use candle::cuda_backend::WrapErr;

        use super::super::migrate::{kv_migrate_on, MigrationPlan};

        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        let cuda_dev = match device {
            candle::Device::Cuda(d) => d,
            _ => {
                return Err(candle::Error::Msg(
                    "migrate_sealed_to_cpu_batch_async requires a CUDA device".into(),
                ))
            }
        };
        let _ = cuda_dev; // referenced only when allocating the staging slice below.

        // ── Resolve sources + dedup ─────────────────────────────────────
        let arena_chunks = GID_STRIDE;
        let arena_info = self.resolve_arena_info()?;
        let mut unique_raws: Vec<i64> = Vec::new();
        let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
        let mut src_ptrs: Vec<(i64, i64)> = Vec::new();
        let mut gid_byte_range: std::collections::HashMap<i64, (usize, usize)> =
            std::collections::HashMap::new();
        let mut src_keys: std::collections::HashMap<i64, ArenaKey> =
            std::collections::HashMap::new();
        let mut total_bytes: usize = 0;

        let elems = self.inner.elems_per_chunk();
        for seq in sequences {
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
                            "migrate_sealed_to_cpu_batch_async: arena {arena_idx} out of range"
                        ))
                    })?;
                    if info.base_ptr == 0 {
                        return Err(candle::Error::Msg(
                            "migrate_sealed_to_cpu_batch_async: source arena not GPU-resident"
                                .into(),
                        ));
                    }
                    // Copy length is the BAND's payload, read from the chunk's
                    // own tag; the address step is the arena's class stride.
                    // The arena cannot supply the first — it holds whatever
                    // fits its slots (`docs/archived/arena_unification.md` invariant 8).
                    let len = payload_bytes_for_tag(tag, elems).ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "migrate_sealed_to_cpu_batch_async: band tag {tag:?} names no                              storage format, so its byte length is unknown"
                        ))
                    })?;
                    let ptr = info.base_ptr as i64 + chunk_idx as i64 * info.chunk_byte_stride;
                    src_ptrs.push((ptr, len as i64));
                    gid_byte_range.insert(raw, (total_bytes, len));
                    total_bytes += len;
                    unique_raws.push(raw);
                    let key = self
                        .inner
                        .storage
                        .read(|s| s.arena_key(arena_idx))?
                        .ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "migrate_sealed_to_cpu_batch_async: arena key {arena_idx} not found"
                            ))
                        })?;
                    src_keys.insert(raw, key);
                }
            }
        }

        if total_bytes == 0 {
            return sequences.iter().map(|seq| Ok((*seq).clone())).collect();
        }

        // ── Ensure pinned scratch capacity ──────────────────────────────
        // **No** `CU_MEMHOSTALLOC_WRITECOMBINED`: this buffer's access
        // pattern is GPU-DtoH-write then **CPU-read** (Phase 4 copies
        // bytes out into the CPU arenas). WC's coalescing helps CPU
        // *writes*; CPU *reads* of WC pages are uncached and slow.
        // The buffer also sees no kernel-I/O fast/slow path issue here,
        // but the cold-load symmetric path uses the same `_default`
        // variant for the same reason — keep the staging buffers'
        // attributes matched to their access pattern.
        let need_grow = pinned_scratch
            .as_ref()
            .map(|b| b.len() < total_bytes)
            .unwrap_or(true);
        if need_grow {
            // Fallible: under host-RAM pressure this returns Err instead of
            // aborting the process (the old `vec![0u8; total_bytes]` fallback
            // called `handle_alloc_error`). A failed migration leaves the turn
            // hot-float + consistent, retried on the next persistence pass.
            *pinned_scratch = Some(PinnedBuf::alloc_default_or_host_fallible(total_bytes)?);
        }

        // ── Phases 1/2: gather → DtoH, in batches the domain can hold ──
        //
        // Staging comes from the persistence domain's transient span rather
        // than the allocator: it is wave-scoped by construction, which is what
        // the bump side is for. **Batched against
        // `MIGRATION_STAGING_CAP_BYTES`, the domain's whole span** — this was a
        // single `bump.alloc(total_bytes)`, so a pass wider than the span failed
        // outright and the constant was a hard ceiling here rather than the
        // batch size its name and doc claimed.
        //
        // One generation per group: dropping it fences the copy stream and
        // rewinds the cursor, so the next group reuses the same bytes safely —
        // the DtoH below has already completed by then (it synchronises).
        let bump = bump_arena::persistence_domain(copy_stream)?;
        let lens: Vec<usize> = src_ptrs.iter().map(|(_, l)| *l as usize).collect();
        let mut group_start = 0usize;
        for (i, j, group_bytes) in staging_groups(&lens, MIGRATION_STAGING_CAP_BYTES) {
            // Allocated through the guard, so `staging` borrows it: the group's
            // range cannot outlive the generation whose drop fences the copy
            // stream and rewinds the cursor under it.
            // No planned layout on the persistence path: its staging is sized per batch
            // rather than by the wave plan, so the whole span stays cursor-managed.
            let group_gen = bump.generation(NOT_A_WAVE, NOT_A_WAVE)?;
            let staging = group_gen.alloc(group_bytes, 256)?;
            let staging_base = staging.ptr as i64;
            let mut plan = MigrationPlan::new();
            let mut off = 0i64;
            for &(ptr, len) in &src_ptrs[i..j] {
                plan.push(ptr, staging_base + off, len);
                off += len;
            }
            kv_migrate_on(device, &plan, Some(copy_stream))?;

            {
                let scratch = pinned_scratch
                    .as_mut()
                    .expect("pinned scratch allocated above");
                let dst = &mut scratch.as_mut_slice()[group_start..group_start + group_bytes];
                // SAFETY: `staging_base` addresses `group_bytes` of the domain's
                // span, held by `_gen` for this scope; `dst` is this group's
                // slice of the pinned host scratch; the sync below completes the
                // copy before either is touched again.
                unsafe {
                    candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async(
                        dst,
                        staging_base as u64,
                        copy_stream.cu_stream(),
                    )
                }
                .map_err(|e| candle::Error::Msg(format!("hot->warm staging readback: {e}")))?;
                copy_stream.synchronize().w()?;
            }

            group_start += group_bytes;
        }

        // ── Phase 3: allocate dest CPU GIDs (one state.write) ──────────
        let mut new_gids: std::collections::HashMap<i64, ChunkGid> =
            std::collections::HashMap::new();
        {
            let _state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for &raw in &unique_raws {
                let src_key = &src_keys[&raw];
                let cpu_key = src_key.at(ArenaLocation::Cpu);
                new_gids.insert(raw, self.alloc_chunk_for_key(cpu_key)?);
            }
        }

        // ── Phase 4: write bytes pinned → CPU arena (one storage.write) ─
        let scratch_ref = pinned_scratch
            .as_ref()
            .expect("pinned scratch allocated above");
        self.inner.storage.try_write(|s| -> candle::Result<()> {
            for &raw in &unique_raws {
                let (off, len) = gid_byte_range[&raw];
                let bytes = &scratch_ref.as_slice()[off..off + len];
                let new_gid = new_gids[&raw].clone();
                Self::write_chunk_from_pinned_bytes(
                    s.arenas_mut(),
                    new_gid.arena_idx(),
                    new_gid.chunk_idx(),
                    bytes,
                )?;
            }
            Ok(())
        })?;

        // ── Phase 5: rebuild SealedSequences with mapped GIDs ──────────
        sequences
            .iter()
            .map(|seq| -> candle::Result<SealedSequence> {
                let new_chunks: candle::Result<Vec<SealedChunk>> = seq
                    .chunks
                    .iter()
                    .map(|chunk| {
                        let mapped = chunk
                            .gids
                            .map_unique(|gid| Ok(new_gids[&gid.raw()].clone()))?;
                        Ok(SealedChunk {
                            gids: mapped,
                            // Demote (GPU→CPU): no device-resident record on the
                            // CPU tier; the chunk uses scratch heads while warm.
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
            .collect()
    }

    /// Migrate one sealed sequence from the warm tier back to the hot tier.
    ///
    /// Symmetric inverse of [`migrate_sealed_to_cpu`].  Sharing structure is
    /// preserved by [`HeadGids::map_unique`].
    ///
    /// The hot tier is the *backing's own device*, not `Gpu` unconditionally: on
    /// a CPU-only backing there is no second tier to lift into, and naming one
    /// used to produce an arena tagged `Gpu` whose bytes were on the host —
    /// harmless while an arena was an ordinary allocation, a contradiction now
    /// that a `Gpu` arena means a region of the device reservation.
    pub fn migrate_sealed_to_gpu(&self, sealed: &SealedSequence) -> candle::Result<SealedSequence> {
        let hot = self.inner.hot_location();
        let gpu_chunks: candle::Result<Vec<SealedChunk>> = sealed
            .chunks
            .iter()
            .map(|chunk| {
                let new_gids = chunk.gids.map_unique(|gid| {
                    let gpu_key = self.inner.storage.read(|s| {
                        s.arena_key(gid.arena_idx())
                            .map(|k| k.at(hot))
                            .ok_or_else(|| {
                                candle::Error::Msg(format!(
                                    "migrate_sealed_to_gpu: arena {} not found",
                                    gid.arena_idx()
                                ))
                            })
                    })??;
                    self.migrate_chunk(gid.raw(), gpu_key)
                })?;
                Ok(SealedChunk {
                    gids: new_gids,
                    // Single-sequence elevate keeps meta=None by design — the
                    // batched elevate (`warm_lift::lift_warm_to_gpu`) is the
                    // one that rebuilds the resident record. This rarely-used path
                    // uses per-forward scratch heads at the new placement.
                    meta: None,
                    ..chunk.clone()
                })
            })
            .collect();
        Ok(SealedSequence {
            chunks: gpu_chunks?,
            token_count: sealed.token_count,
            chunk_size: sealed.chunk_size,
            location: hot,
        })
    }
}
