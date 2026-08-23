//! VRAM↔RAM transfer orchestration (§3, §4, §16.6 of
//! `docs/kv_tier_migration.md`).
//!
//! `transfer.rs` drives `candle-nn`'s `kv_migrate` scatter/gather kernel:
//!
//! - [`gather_chunks`] — resolve a set of scattered VRAM chunks into one
//!   contiguous host buffer (the core of eviction).
//! - [`scatter_chunks`] — write a contiguous host buffer back into a set of
//!   VRAM chunks (the core of a load).
//! - [`load_to_hot`] — the `SealedSequence`-level load path: thin glue
//!   over the two primitives that materialises a recovered chunk grid
//!   into a fresh per-layer `SealedSequence` set in VRAM.
//!
//! `candle-conversation` is CUDA-only, so this module is unconditionally
//! compiled — the gather/scatter helpers and `kv_migrate` plan it builds
//! all assume a CUDA device.

mod cuda_impl {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::{Device, Result};
    use candle_nn::kv_cache::{
        fletcher32_golden, kv_migrate, ArenaLocation, ChunkedKvBacking, GoldenRecord,
        MigrationPlan, SealedSequence,
    };

    use crate::persistence::cold_load::{ColdLoadStager, PINNED_PREALLOC_BYTES};
    use crate::persistence::resume::ChunkImage;
    use crate::persistence::streams::{StreamId, TurnDecl};
    use crate::persistence::SubstratePersistence;
    use crate::substrate::Substrate;

    fn cuda_device(device: &Device) -> Result<&candle::CudaDevice> {
        match device {
            Device::Cuda(d) => Ok(d),
            _ => Err(candle::Error::Msg("transfer requires a CUDA device".into())),
        }
    }

    /// Gather scattered VRAM `chunks` — `(device_ptr, byte_len)` pairs — into
    /// one contiguous host buffer, in order. This is `kv_pack` followed by a
    /// device-to-host copy.
    pub fn gather_chunks(device: &Device, chunks: &[(i64, i64)]) -> Result<Vec<u8>> {
        let dev = cuda_device(device)?;
        let total: i64 = chunks.iter().map(|&(_, len)| len).sum();
        if total == 0 {
            return Ok(Vec::new());
        }

        let staging = unsafe {
            dev.alloc::<u8>(total as usize)
                .map_err(|e| candle::Error::Msg(format!("transfer: staging alloc: {e}")))?
        };
        let staging_base = {
            let stream = dev.cuda_stream();
            let base = staging.device_ptr(&stream).0 as i64;
            base
        };

        let mut plan = MigrationPlan::new();
        let mut offset = 0i64;
        for &(ptr, len) in chunks {
            plan.push(ptr, staging_base + offset, len);
            offset += len;
        }
        kv_migrate(device, &plan)?;

        dev.memcpy_dtov(&staging)
            .map_err(|e| candle::Error::Msg(format!("transfer: staging DtoH: {e}")))
    }

    /// Gather like [`gather_chunks`], but also compute a Fletcher-32 **golden**
    /// per output chunk on the GPU — over the staging buffer, *before* the DtoH
    /// copy. `split_sizes` partitions the gathered blob into the per-`ChunkImage`
    /// KV spans (each `SealedChunk.byte_size`); their sum must equal the total
    /// gathered byte count. Returns the host blob and one golden per split, in
    /// order.
    ///
    /// Computing the golden on the device here — not host-side after the copy —
    /// is the point: it captures the bytes exactly as the GPU produced them, so
    /// a later CPU recompute over the warm/cold copy (at reload) detects
    /// corruption introduced by the DtoH copy or on storage.
    pub fn gather_chunks_with_goldens(
        device: &Device,
        chunks: &[(i64, i64)],
        split_sizes: &[usize],
    ) -> Result<(Vec<u8>, Vec<u32>)> {
        let dev = cuda_device(device)?;
        let total: i64 = chunks.iter().map(|&(_, len)| len).sum();
        let split_total: usize = split_sizes.iter().sum();
        if split_total as i64 != total {
            return Err(candle::Error::Msg(format!(
                "gather_chunks_with_goldens: splits sum to {split_total}, blob is {total} bytes"
            )));
        }
        if total == 0 {
            return Ok((Vec::new(), vec![0u32; split_sizes.len()]));
        }

        let staging = unsafe {
            dev.alloc::<u8>(total as usize)
                .map_err(|e| candle::Error::Msg(format!("transfer: staging alloc: {e}")))?
        };
        let staging_base = {
            let stream = dev.cuda_stream();
            let base = staging.device_ptr(&stream).0 as i64;
            base
        };

        let mut plan = MigrationPlan::new();
        let mut offset = 0i64;
        for &(ptr, len) in chunks {
            plan.push(ptr, staging_base + offset, len);
            offset += len;
        }
        kv_migrate(device, &plan)?;

        // Golden per output chunk, over the staging buffer, before the DtoH copy.
        let mut records = Vec::with_capacity(split_sizes.len());
        let mut goff = 0i64;
        for &n in split_sizes {
            records.push(GoldenRecord {
                src_ptr: staging_base + goff,
                byte_len: n as i64,
            });
            goff += n as i64;
        }
        let goldens = fletcher32_golden(device, &records)?;

        let blob = dev
            .memcpy_dtov(&staging)
            .map_err(|e| candle::Error::Msg(format!("transfer: staging DtoH: {e}")))?;
        Ok((blob, goldens))
    }

    /// Scatter a contiguous host buffer back into scattered VRAM `chunks`.
    /// This is a host-to-device copy followed by `kv_unpack`. `host` must be
    /// exactly the summed `byte_len` of `chunks`.
    pub fn scatter_chunks(device: &Device, chunks: &[(i64, i64)], host: &[u8]) -> Result<()> {
        let dev = cuda_device(device)?;
        let total: i64 = chunks.iter().map(|&(_, len)| len).sum();
        if host.len() as i64 != total {
            return Err(candle::Error::Msg(format!(
                "transfer: scatter host buffer is {} bytes, chunks need {total}",
                host.len()
            )));
        }
        if total == 0 {
            return Ok(());
        }

        let staging = dev
            .memcpy_stod(host)
            .map_err(|e| candle::Error::Msg(format!("transfer: staging HtoD: {e}")))?;
        let staging_base = {
            let stream = dev.cuda_stream();
            let base = staging.device_ptr(&stream).0 as i64;
            base
        };

        let mut plan = MigrationPlan::new();
        let mut offset = 0i64;
        for &(ptr, len) in chunks {
            plan.push(staging_base + offset, ptr, len);
            offset += len;
        }
        kv_migrate(device, &plan)
    }

    /// Gather a sealed sequence off the GPU into the realloc-able
    /// [`ChunkImage`] grid — `resolve_sealed_chunk_ptrs` + `gather_chunks`
    /// into one opaque blob, split per chunk by `SealedChunk.byte_size` and
    /// paired with each chunk's offset, K/V formats, palettes and scales.
    ///
    /// This is the shared gather used by both warm eviction and the
    /// seal-time redo-log write — the representation the cold-load path
    /// rebuilds VRAM chunks from.
    pub fn seal_to_chunk_images(
        backing: &ChunkedKvBacking,
        device: &Device,
        seq: &SealedSequence,
    ) -> Result<Vec<ChunkImage>> {
        match seq.location {
            ArenaLocation::Gpu => seal_to_chunk_images_gpu(backing, device, seq),
            ArenaLocation::Cpu => seal_to_chunk_images_cpu(backing, seq),
        }
    }

    /// GPU gather — the fast path. One `kv_migrate_on` gather + one
    /// `cudaMemcpyDtoH` for the whole sealed sequence.
    fn seal_to_chunk_images_gpu(
        backing: &ChunkedKvBacking,
        device: &Device,
        seq: &SealedSequence,
    ) -> Result<Vec<ChunkImage>> {
        use crate::persistence::record::ChunkPayload;

        let ptrs = backing.resolve_sealed_chunk_ptrs(&seq.chunks)?;
        let split_sizes: Vec<usize> = seq.chunks.iter().map(|sc| sc.byte_size as usize).collect();
        // Golden computed on-GPU over the staging buffer before the DtoH copy —
        // `goldens[i]` is the Fletcher-32 of chunk `i`'s KV bytes as the device
        // produced them, the ground truth the reload verifies the on-disk copy against.
        let (blob, goldens) = gather_chunks_with_goldens(device, &ptrs, &split_sizes)?;
        let mut cursor = 0usize;
        let mut images = Vec::with_capacity(seq.chunks.len());
        for (i, sc) in seq.chunks.iter().enumerate() {
            let n = sc.byte_size as usize;
            if cursor + n > blob.len() {
                return Err(candle::Error::Msg(format!(
                    "seal_to_chunk_images: blob underrun (need {n} at {cursor}, have {})",
                    blob.len()
                )));
            }
            let kv_bytes = blob[cursor..cursor + n].to_vec();
            cursor += n;
            let (k_formats, v_formats) = sc.format_tags()?;
            images.push(ChunkImage {
                token_count: sc.token_count,
                golden: goldens[i],
                payload: ChunkPayload {
                    offset: sc.offset,
                    k_formats: k_formats.to_vec(),
                    v_formats: v_formats.to_vec(),
                    k_pal: (*sc.k_pal).clone(),
                    v_pal: (*sc.v_pal).clone(),
                    k_scale: (*sc.k_scale).clone(),
                    v_scale: (*sc.v_scale).clone(),
                    kv_bytes,
                },
            });
        }
        Ok(images)
    }

    /// CPU gather — reads each chunk's bytes directly out of the
    /// CPU arena via `read_chunk_into_bytes`. Used when warm holds
    /// quantize-on-evict outputs that must be persisted to cold.
    /// Slower per-byte than the GPU gather (no batched DMA) but the
    /// persist path is off the hot loop.
    fn seal_to_chunk_images_cpu(
        backing: &ChunkedKvBacking,
        seq: &SealedSequence,
    ) -> Result<Vec<ChunkImage>> {
        use candle_nn::kv_cache::{payload_bytes_for_tag, GID_STRIDE};

        use crate::persistence::record::ChunkPayload;

        let arena_chunks = GID_STRIDE;
        let arena_info = backing.resolve_arena_info()?;
        let elems_per_chunk = backing.elems_per_chunk();
        // Pull each unique (arena, chunk) slot's bytes into a single
        // blob, then split per `sc.byte_size` like the GPU path. This
        // preserves the SealedChunk's per-chunk byte_size accounting
        // exactly, including the per-(h, p) GID dedup that
        // `arena_byte_size` already computed.
        let mut seen = std::collections::HashSet::new();
        let mut blob: Vec<u8> = Vec::new();
        for chunk in &seq.chunks {
            for (gid, tag) in chunk.bands() {
                let raw = gid.raw();
                if !seen.insert(raw) {
                    continue;
                }
                let arena_idx = (raw as usize) / arena_chunks;
                let chunk_idx = (raw as usize) % arena_chunks;
                if arena_info.get(arena_idx).is_none() {
                    return Err(candle::Error::Msg(format!(
                        "seal_to_chunk_images_cpu: arena_idx {arena_idx} out of range"
                    )));
                }
                // Reserve the band's PAYLOAD, not its slot stride: the blob is
                // split below by `sc.byte_size`, which sums payloads. Reserving
                // the stride would desynchronise the two the moment a class
                // slot exceeds the format's bytes — `blob underrun` at best,
                // silently shifted chunk boundaries in persisted data at worst.
                //
                // The payload comes from the band's own tag, because the arena
                // is a run of untyped byte slots and has no format to report
                // (`docs/archived/arena_unification.md` invariant 8).
                let payload = payload_bytes_for_tag(tag, elems_per_chunk).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "seal_to_chunk_images_cpu: band tag {tag:?} names no storage format, \
                         so its byte length is unknown"
                    ))
                })?;
                let start = blob.len();
                blob.resize(start + payload, 0);
                backing.read_chunk_into_bytes(
                    arena_idx,
                    chunk_idx,
                    &mut blob[start..start + payload],
                )?;
            }
        }
        let mut cursor = 0usize;
        let mut images = Vec::with_capacity(seq.chunks.len());
        for sc in &seq.chunks {
            let n = sc.byte_size as usize;
            if cursor + n > blob.len() {
                return Err(candle::Error::Msg(format!(
                    "seal_to_chunk_images_cpu: blob underrun (need {n} at {cursor}, have {})",
                    blob.len()
                )));
            }
            let kv_bytes = blob[cursor..cursor + n].to_vec();
            cursor += n;
            // The CPU gather reads the arena bytes directly host-side (no GPU
            // round trip), so a host-side Fletcher is itself the ground truth —
            // there is no DtoH copy between the read and this checksum.
            let golden = candle::fletcher::fletcher32(&kv_bytes);
            let (k_formats, v_formats) = sc.format_tags()?;
            images.push(ChunkImage {
                token_count: sc.token_count,
                golden,
                payload: ChunkPayload {
                    offset: sc.offset,
                    k_formats: k_formats.to_vec(),
                    v_formats: v_formats.to_vec(),
                    k_pal: (*sc.k_pal).clone(),
                    v_pal: (*sc.v_pal).clone(),
                    k_scale: (*sc.k_scale).clone(),
                    v_scale: (*sc.v_scale).clone(),
                    kv_bytes,
                },
            });
        }
        Ok(images)
    }

    /// Cold-load fast path — stream a turn's records through a
    /// fixed-size pinned scratch and migrate them onto the GPU.
    ///
    /// The turn's records are partitioned by
    /// [`crate::persistence::chunk_plan::plan_chunked_read`] into chunk
    /// batches that each fit within the stager's pinned buffer. Each
    /// batch then runs through [`crate::persistence::pipeline::run_pipeline`],
    /// which fans the batch out into 1 MiB units processed concurrently
    /// by a reader pool, a per-layer bulk allocator, and a GPU
    /// dispatcher — reads, HtoDs, and the `kv_migrate` scatter all
    /// overlap at sub-stripe granularity.
    ///
    /// After every batch has been processed, each layer's final
    /// `record_turn` produces the `SealedSequence` returned to the
    /// caller. Memory bound: one [`PINNED_PREALLOC_BYTES`]-sized
    /// pinned buffer + one same-sized GPU staging slice per batch —
    /// independent of turn size.
    ///
    /// Returns one `SealedSequence` per layer; the order matches `backings`.
    pub fn load_turn_into_hot(
        backings: &[ChunkedKvBacking],
        device: &Device,
        persistence: &mut SubstratePersistence,
        substrate: &Substrate,
        decl: &TurnDecl,
        stager: &mut ColdLoadStager,
    ) -> Result<Vec<SealedSequence>> {
        let chunks_per_layer = (decl.block_end - decl.block_start) as usize;
        let stream_id =
            crate::persistence::content_hash::turn_stream_id(decl.timeline_id, decl.turn_index);
        load_stream_into_hot(
            backings,
            device,
            persistence,
            substrate,
            stream_id,
            chunks_per_layer,
            stager,
        )
    }

    /// Load any persisted stream (turn or section) from cold storage
    /// into hot VRAM.  Generalised body of [`load_turn_into_hot`];
    /// section paths call this directly with their content-addressed
    /// stream id + `manifest.chunks.len() / n_layers` as
    /// `chunks_per_layer`.  The behaviour and timing are identical —
    /// the routing is just a parameterisation difference.
    pub fn load_stream_into_hot(
        backings: &[ChunkedKvBacking],
        device: &Device,
        persistence: &mut SubstratePersistence,
        substrate: &Substrate,
        stream_id: StreamId,
        chunks_per_layer: usize,
        stager: &mut ColdLoadStager,
    ) -> Result<Vec<SealedSequence>> {
        use std::time::Instant;

        let n_layers = backings.len();
        // Early CUDA-device validation; the pipeline itself re-derives
        // the device for its own use.
        let _dev = cuda_device(device)?;

        let t_total = Instant::now();

        // Build the chunk plan up front, sized to the stager's ACTUAL pinned
        // buffer so large turns (>64 MB) partition into batches that each fit —
        // the `for batch in &plan.chunks` loop below streams them. `0` means the
        // buffer is not yet allocated (unit-test `::new()` path): plan for the
        // prealloc size the first `buffer_mut()` will lazily allocate.
        //
        // NB: the previous `.max(1 << 30)` planned 1 GB batches against a fixed
        // 64 MB buffer, so any turn over 64 MB became one oversized batch and
        // overflowed `buffer_mut` (the chunked-read assert).
        let buffer_size = match stager.capacity() {
            0 => PINNED_PREALLOC_BYTES,
            cap => cap,
        };
        let plan = persistence.plan_chunked_read(substrate, stream_id, buffer_size);

        // Empty turn — no chunks. Build empty sealed sequences per
        // layer (matches the `recover_turn_grid` + `load_to_hot`
        // pair on an empty grid).
        if plan.n_records == 0 {
            let mut out = Vec::with_capacity(n_layers);
            for backing in backings {
                let slot = backing.alloc_sequence()?;
                let seq = backing.record_turn(slot)?;
                backing.free_sequence(slot)?;
                out.push(seq);
            }
            return Ok(out);
        }

        // RAII: free every scratch slot on drop, success or error.
        struct SlotsGuard<'a> {
            backings: &'a [ChunkedKvBacking],
            items: Vec<(usize, usize)>,
        }
        impl<'a> Drop for SlotsGuard<'a> {
            fn drop(&mut self) {
                for &(bi, slot) in &self.items {
                    let _ = self.backings[bi].free_sequence(slot);
                }
            }
        }
        let mut slots_guard = SlotsGuard {
            backings,
            items: Vec::with_capacity(n_layers),
        };

        // Pre-allocate one slot per layer. Holds the accumulated
        // chunks across every chunk-batch's reads.
        let mut slots: Vec<usize> = Vec::with_capacity(n_layers);
        for (li, backing) in backings.iter().enumerate() {
            let slot = backing.alloc_sequence()?;
            slots_guard.items.push((li, slot));
            slots.push(slot);
        }

        // Per-chunk-batch loop: each batch fans out into the 1 MiB-unit
        // pipeline (reader pool, allocator thread, GPU dispatcher on
        // main thread). Reads, HtoDs, and the migrate kernel all
        // overlap at sub-stripe granularity.
        let mut reads_ms_total: u64 = 0;
        let mut alloc_ms_total: u64 = 0;
        let mut htod_ms_total: u64 = 0;
        let mut migrate_ms_total: u64 = 0;
        let mut n_units_total: u32 = 0;
        // Allocator sub-bucket breakdown (microseconds).
        let mut decode_us_total: u64 = 0;
        let mut bulk_alloc_us_total: u64 = 0;
        let mut resolve_ptrs_us_total: u64 = 0;
        let mut n_bulk_calls_total: u32 = 0;
        let mut n_resolve_calls_total: u32 = 0;
        let mut pool_us_total: u64 = 0;
        let mut register_us_total: u64 = 0;
        let mut gpu_push_us_total: u64 = 0;

        // Open a direct-I/O handle set for every sealed segment this turn's
        // chunks span (a turn that straddles a seal reads part of its records
        // from a sealed segment). The active and inherited handles come from
        // `persistence`; the sealed ones are owned here for the cold-load's
        // duration so the pipeline's reader threads can borrow them by id.
        let mut sealed_handles: std::collections::HashMap<
            crate::persistence::segment::SegmentId,
            candle::direct_io::DirectFile,
        > = std::collections::HashMap::new();
        for batch in &plan.chunks {
            if let crate::persistence::chunk_plan::SourceLog::Sealed(id) = batch.source {
                if let std::collections::hash_map::Entry::Vacant(e) = sealed_handles.entry(id) {
                    let handle = persistence.open_sealed_direct(id).map_err(|e| {
                        candle::Error::Msg(format!("cold-load: open sealed segment {id}: {e}"))
                    })?;
                    e.insert(handle);
                }
            }
        }

        for batch in &plan.chunks {
            let stats = crate::persistence::pipeline::run_pipeline(
                persistence,
                &sealed_handles,
                backings,
                device,
                batch,
                stager,
                &slots,
                chunks_per_layer,
            )?;
            reads_ms_total += stats.reads_ms;
            alloc_ms_total += stats.alloc_ms;
            htod_ms_total += stats.htod_ms;
            migrate_ms_total += stats.migrate_ms;
            n_units_total += stats.n_units;
            decode_us_total += stats.decode_us;
            bulk_alloc_us_total += stats.bulk_alloc_us;
            resolve_ptrs_us_total += stats.resolve_ptrs_us;
            n_bulk_calls_total += stats.n_bulk_calls;
            n_resolve_calls_total += stats.n_resolve_calls;
            pool_us_total += stats.pool_us;
            register_us_total += stats.register_us;
            gpu_push_us_total += stats.gpu_push_us;
        }

        // Final per-layer SealedSequence — the snapshot caller wanted.
        let mut sealed_per_layer: Vec<SealedSequence> = Vec::with_capacity(n_layers);
        for (li, backing) in backings.iter().enumerate() {
            let seq = backing.record_turn(slots[li])?;
            sealed_per_layer.push(seq);
        }

        let total_ms = t_total.elapsed().as_millis();
        tracing::debug!(
            target: "candle_conversation::persistence::tier",
            stream_id = stream_id.0,
            n_chunks_total = plan.n_records,
            pinned_bytes = plan.total_bytes,
            n_chunk_batches = plan.chunks.len(),
            n_units = n_units_total,
            reads_ms = reads_ms_total,
            alloc_ms = alloc_ms_total,
            htod_ms = htod_ms_total,
            migrate_ms = migrate_ms_total,
            total_ms,
            "load_stream_into_hot timing (pipelined)"
        );

        // Separate breakdown line for the allocator stage so the
        // top-line "load_stream_into_hot timing" stays compact. The
        // three sub-timers add up to ~`alloc_ms` (in µs); the gap
        // is per-unit bookkeeping (`per_layer` Vec construction,
        // `records_to_dispatch` builds, channel sends). Emitted at
        // trace level since it's only useful when actively profiling
        // the alloc path — enable with
        // `RUST_LOG=candle_conversation::persistence::tier=trace`.
        tracing::trace!(
            target: "candle_conversation::persistence::tier",
            stream_id = stream_id.0,
            alloc_ms = alloc_ms_total,
            decode_us = decode_us_total,
            bulk_alloc_us = bulk_alloc_us_total,
            pool_us = pool_us_total,
            register_us = register_us_total,
            gpu_push_us = gpu_push_us_total,
            resolve_ptrs_us = resolve_ptrs_us_total,
            n_bulk_calls = n_bulk_calls_total,
            n_resolve_calls = n_resolve_calls_total,
            "load_turn_into_hot alloc breakdown"
        );

        Ok(sealed_per_layer)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // ── Forensic decoder for the persist path (V-corruption hunt) ──
        //
        // Reproduces decode → `quantize_sealed_in_place` (hot→warm) →
        // `seal_to_chunk_images` (warm→cold CPU gather) and dequantizes
        // the gathered `ChunkPayload` back to `[h][d][t]` so we can cosine
        // K and V against the distinct float patterns we seeded. If V comes
        // back orthogonal (≈0) while K survives, the CPU gather corrupts V.
        use candle::quantized::{GgmlDType, QStorage, QTensor};
        use candle::{DType, Tensor};
        use candle_nn::kv_cache::{
            quantize_sealed_in_place, CompressionPolicy, KvFormat, QuantFormat, GID_STRIDE,
            N_PALETTE,
        };
        use half::f16;

        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        use crate::persistence::record::ChunkPayload;

        const FZ_N_KV_HEAD: usize = 4;
        const FZ_HEAD_DIM: usize = 128;
        const FZ_ARENA_CAPACITY: usize = 256;

        fn cuda_device_or_skip() -> Option<candle::Device> {
            match candle::Device::cuda_if_available(0) {
                Ok(d @ candle::Device::Cuda(_)) => Some(d),
                _ => None,
            }
        }

        fn cuda_stream(device: &candle::Device) -> Arc<CudaStream> {
            match device {
                candle::Device::Cuda(d) => d.cuda_stream(),
                _ => unreachable!("gated on a CUDA device"),
            }
        }

        /// Resolve a persisted `KvFormat` tag to its GGML block dtype.
        /// Errors on a Float band — C1 seals both K and V quantized here.
        fn kv_tag_to_ggml(tag: u8) -> Result<GgmlDType> {
            let fmt = KvFormat::from_tag(tag)
                .ok_or_else(|| candle::Error::Msg(format!("unrecognised KvFormat tag {tag}")))?;
            match fmt {
                KvFormat::Quantized(qf) => Ok(qf.to_ggml_dtype()),
                KvFormat::Float(dt) => Err(candle::Error::Msg(format!(
                    "sub-band is Float({dt:?}) (tag {tag}); this decoder handles \
                     quantized sub-bands only"
                ))),
            }
        }

        /// Bytes one sub-band (`band_elems` elements) occupies for `ggml`.
        fn band_byte_size(ggml: GgmlDType, band_elems: usize) -> usize {
            (band_elems / ggml.block_size()) * ggml.type_size()
        }

        /// Dequantize one palette sub-band's raw arena bytes to a
        /// `sub_head_dim * chunk_size` f32 vector in **dim-major** `[pd][t]`
        /// order — the layout the quant/R16 arenas physically store (one
        /// 32-token block per head-dimension). Index as `band[pd*chunk_size+t]`.
        fn dequant_sub_band(
            ggml: GgmlDType,
            raw: &[u8],
            chunk_size: usize,
            sub_head_dim: usize,
        ) -> Result<Vec<f32>> {
            let elem_count = chunk_size * sub_head_dim;
            let blocks = ggml.cpu_zeros(elem_count);
            let need = blocks.storage_size_in_bytes();
            if raw.len() != need {
                return Err(candle::Error::Msg(format!(
                    "{ggml:?} sub-band: raw {} bytes != block storage {need} bytes",
                    raw.len()
                )));
            }
            let dst = blocks.as_ptr() as *mut u8;
            // SAFETY: `dst` owns `need` bytes; `raw` is a disjoint slice of
            // exactly `need` bytes; block types are POD.
            unsafe {
                std::ptr::copy_nonoverlapping(raw.as_ptr(), dst, need);
            }
            let storage = QStorage::Cpu(blocks);
            let qt = QTensor::new(storage, vec![chunk_size, sub_head_dim])?;
            let t = qt.dequantize(&candle::Device::Cpu)?;
            t.flatten_all()?.to_vec1::<f32>()
        }

        /// Reassemble one block from its `ChunkPayload`, walking `kv_bytes`
        /// in interleaved arena-GID order `[K(h,p) V(h,p) …]` (h outer, p
        /// inner) and dequantizing each K/V sub-band into the per-layer
        /// `[n_kv_head][head_dim][chunk_size]` layout
        /// (`out[(h*head_dim + d)*chunk_size + t]`).
        fn reassemble_block(
            payload: &ChunkPayload,
            n_kv_head: usize,
            head_dim: usize,
            chunk_size: usize,
            sub_head_dim: usize,
            per_layer: usize,
        ) -> Result<(Vec<f32>, Vec<f32>)> {
            let band_elems = chunk_size * sub_head_dim;
            let expect_bands = n_kv_head * N_PALETTE;
            if payload.k_formats.len() != expect_bands || payload.v_formats.len() != expect_bands {
                return Err(candle::Error::Msg(format!(
                    "payload has {}/{} K/V sub-bands, expected {expect_bands}",
                    payload.k_formats.len(),
                    payload.v_formats.len(),
                )));
            }
            let mut k_out = vec![0.0f32; per_layer];
            let mut v_out = vec![0.0f32; per_layer];
            let mut cursor = 0usize;
            let blob = &payload.kv_bytes;
            let pal_bytes = head_dim / 4;
            // Palette routing: per head, the ordered global dims each palette
            // owns. Sub-band `p`'s local dim `pd` → `pal_dims[p][pd]`. Empty pal
            // map (identity routing) falls back to contiguous `p*sub + pd`.
            let dims_for = |pal: &[u8], head: usize| -> [Vec<usize>; N_PALETTE] {
                let mut out: [Vec<usize>; N_PALETTE] = Default::default();
                if pal.len() < (head + 1) * pal_bytes {
                    for d in 0..head_dim {
                        out[d / sub_head_dim].push(d);
                    }
                } else {
                    for d in 0..head_dim {
                        let byte = pal[head * pal_bytes + d / 4];
                        let p = ((byte >> (2 * (d % 4))) & 0x3) as usize;
                        out[p].push(d);
                    }
                }
                out
            };
            for h in 0..n_kv_head {
                let k_dims = dims_for(&payload.k_pal, h);
                let v_dims = dims_for(&payload.v_pal, h);
                for p in 0..N_PALETTE {
                    let sb = h * N_PALETTE + p;
                    let k_ggml = kv_tag_to_ggml(payload.k_formats[sb])?;
                    let k_bytes = band_byte_size(k_ggml, band_elems);
                    if cursor + k_bytes > blob.len() {
                        return Err(candle::Error::Msg(format!(
                            "kv_bytes underrun at K(h={h},p={p}): need {k_bytes} at {cursor}, have {}",
                            blob.len()
                        )));
                    }
                    let k_band = dequant_sub_band(
                        k_ggml,
                        &blob[cursor..cursor + k_bytes],
                        chunk_size,
                        sub_head_dim,
                    )?;
                    let k_off = cursor;
                    cursor += k_bytes;

                    let v_ggml = kv_tag_to_ggml(payload.v_formats[sb])?;
                    let v_bytes = band_byte_size(v_ggml, band_elems);
                    if cursor + v_bytes > blob.len() {
                        return Err(candle::Error::Msg(format!(
                            "kv_bytes underrun at V(h={h},p={p}): need {v_bytes} at {cursor}, have {}",
                            blob.len()
                        )));
                    }
                    let v_band = dequant_sub_band(
                        v_ggml,
                        &blob[cursor..cursor + v_bytes],
                        chunk_size,
                        sub_head_dim,
                    )?;
                    let v_off = cursor;
                    cursor += v_bytes;

                    // Only noisy when hunting the bug — per-sub-band offsets
                    // help localise a K/V swap or a misaligned V band.
                    if std::env::var("FZ_TRACE").is_ok() {
                        eprintln!(
                            "  sb {sb:>2} (h={h},p={p})  K@{k_off} {k_ggml:?} ({k_bytes}B)  \
                             V@{v_off} {v_ggml:?} ({v_bytes}B)"
                        );
                    }

                    // Bands are dim-major (`band[pd*chunk_size+t]`); route each
                    // local dim `pd` to its global head dim via the palette map.
                    for (pd, &kd) in k_dims[p].iter().take(sub_head_dim).enumerate() {
                        for t in 0..chunk_size {
                            k_out[(h * head_dim + kd) * chunk_size + t] =
                                k_band[pd * chunk_size + t];
                        }
                    }
                    for (pd, &vd) in v_dims[p].iter().take(sub_head_dim).enumerate() {
                        for t in 0..chunk_size {
                            v_out[(h * head_dim + vd) * chunk_size + t] =
                                v_band[pd * chunk_size + t];
                        }
                    }
                }
            }
            Ok((k_out, v_out))
        }

        fn cosine(a: &[f32], b: &[f32]) -> f32 {
            let n = a.len().min(b.len());
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for i in 0..n {
                let (x, y) = (a[i] as f64, b[i] as f64);
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            if na == 0.0 || nb == 0.0 {
                return 0.0;
            }
            (dot / (na.sqrt() * nb.sqrt())) as f32
        }

        /// The PERSIST path, end to end, on synthetic K/V:
        ///
        ///   decode float write → `quantize_sealed_in_place` (hot→warm) →
        ///   `seal_to_chunk_images` (warm→cold CPU gather).
        ///
        /// K and V are seeded with **distinct** float patterns (different
        /// sign and slope) so a K/V swap or a V-orthogonal corruption in
        /// the gather is visible in the cosine. We dequantize block 0's
        /// gathered payload and cosine-compare against the seeded inputs.
        ///
        /// K validates the decoder + `[h][d][t]` transpose (must be >0.9).
        /// Only then is the V verdict trustworthy.
        #[test]
        fn persist_gather_preserves_v_vectors() {
            let Some(device) = cuda_device_or_skip() else {
                eprintln!("no CUDA device — skipping persist_gather_preserves_v_vectors");
                return;
            };
            let n_kv_head = FZ_N_KV_HEAD;
            let head_dim = FZ_HEAD_DIM;
            let n_tokens = 32usize; // one full 32-token block
            let chunk_size = candle_nn::CHUNK_SIZE;
            let sub_head_dim = head_dim / N_PALETTE;
            let per_layer = n_kv_head * head_dim * chunk_size;
            assert_eq!(n_tokens, chunk_size, "one full block");

            // C1 — production recall level: K→Q8_KS, V→Q4_0/Q8_0.
            let policy = CompressionPolicy::new(1);

            // `new_with_format_adaptive(..., Some(policy))` warms the shared
            // adaptive candidate arenas at construction — the public path the
            // substrate engine uses (the internal `warm_protected_arenas` is
            // `pub(super)` to candle-nn).
            let backing = ChunkedKvBacking::new_with_format_adaptive(
                4,
                n_kv_head,
                head_dim,
                KvFormat::Float(DType::F16),
                KvFormat::Float(DType::F16),
                &device,
                FZ_ARENA_CAPACITY,
                Some(policy),
            )
            .expect("create chunked backing with warmed candidate arenas");

            // ── Seed distinct K and V float patterns ──────────────────
            // Input layout from `write_contiguous((1, n_kv_head, n_tokens,
            // head_dim))`: element (h,t,d) at ((h*n_tokens)+t)*head_dim + d.
            let slot = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
            let total = n_kv_head * n_tokens * head_dim;
            let k_data: Vec<f16> = (0..total)
                .map(|i| f16::from_f32(((1000 + i) as f32) * 0.0005))
                .collect();
            let v_data: Vec<f16> = (0..total)
                .map(|i| f16::from_f32(-((1000 + i) as f32) * 0.0007))
                .collect();
            let k = Tensor::from_vec(
                k_data.clone(),
                (1, n_kv_head, n_tokens, head_dim),
                &candle::Device::Cpu,
            )
            .unwrap()
            .to_device(&device)
            .unwrap();
            let v = Tensor::from_vec(
                v_data.clone(),
                (1, n_kv_head, n_tokens, head_dim),
                &candle::Device::Cpu,
            )
            .unwrap()
            .to_device(&device)
            .unwrap();
            backing.write_contiguous(slot, 0, &k, &v).unwrap();
            backing.set_len(slot, n_tokens);
            let src = backing.record_turn(slot).unwrap();

            // Reindex the inputs into the reassembled `[h][d][t]` layout so
            // the cosine compares matched element sets.
            let idx_in = |h: usize, t: usize, d: usize| ((h * n_tokens) + t) * head_dim + d;
            let idx_out = |h: usize, d: usize, t: usize| (h * head_dim + d) * chunk_size + t;
            let mut k_in = vec![0.0f32; per_layer];
            let mut v_in = vec![0.0f32; per_layer];
            for h in 0..n_kv_head {
                for d in 0..head_dim {
                    for t in 0..n_tokens {
                        k_in[idx_out(h, d, t)] = k_data[idx_in(h, t, d)].to_f32();
                        v_in[idx_out(h, d, t)] = v_data[idx_in(h, t, d)].to_f32();
                    }
                }
            }

            // ── Step 1: hot→warm quantize (CPU-resident output) ───────
            let copy_stream = cuda_stream(&device);
            let mut pinned: Option<PinnedBuf> = None;
            let warm = quantize_sealed_in_place(
                &backing,
                &[&src],
                &policy,
                &device,
                &copy_stream,
                &mut pinned,
            )
            .expect("quantize_sealed_in_place");
            assert_eq!(warm.len(), 1);
            let warm = &warm[0];

            // ── Step 2: warm→cold CPU gather ──────────────────────────
            // `seal_to_chunk_images` dispatches to the CPU gather when the
            // sequence is CPU-resident (the recall persist path). If warm
            // stayed on GPU, migrate it to CPU so we exercise the CPU gather
            // (the suspected corrupting step).
            let _stride = GID_STRIDE;
            let cpu_seq;
            let seq_ref: &SealedSequence = if warm.location == ArenaLocation::Cpu {
                warm
            } else {
                let migrated = backing
                    .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[warm])
                    .expect("migrate warm → CPU for the cold gather");
                cpu_seq = migrated.into_iter().next().unwrap();
                &cpu_seq
            };
            assert_eq!(
                seq_ref.location,
                ArenaLocation::Cpu,
                "gather must run over a CPU-resident sequence (cold write path)"
            );

            let images =
                seal_to_chunk_images(&backing, &device, seq_ref).expect("seal_to_chunk_images");
            assert!(!images.is_empty(), "expected at least one gathered block");
            let payload = &images[0].payload;

            // ── Dequantize + cosine ───────────────────────────────────
            let (k_rt, v_rt) = reassemble_block(
                payload,
                n_kv_head,
                head_dim,
                chunk_size,
                sub_head_dim,
                per_layer,
            )
            .expect("reassemble block 0");

            let k_cos = cosine(&k_rt, &k_in);
            let v_cos = cosine(&v_rt, &v_in);
            let v_vs_k = cosine(&v_rt, &k_in); // detect a K/V swap
            eprintln!("PERSIST-GATHER  K cosine = {k_cos:.5}   V cosine = {v_cos:.5}   (V-vs-inputK = {v_vs_k:.5})");

            // K validates the decoder + layout transpose. Do not trust V
            // until K is high.
            assert!(
                k_cos > 0.9,
                "K cosine {k_cos:.5} <= 0.9 — decoder/layout/transpose is wrong; \
                 fix the input↔[h][d][t] transpose before trusting V"
            );

            if v_cos > 0.9 {
                eprintln!(
                    "VERDICT: gather is CLEAN here — V survives the persist path \
                     (bug NOT reproduced by this synthetic path)."
                );
                assert!(
                    v_cos > 0.9,
                    "V cosine {v_cos:.5} — documents that the CPU gather preserves V"
                );
            } else if v_cos.abs() <= 0.30 {
                eprintln!(
                    "VERDICT: BUG REPRODUCED — the CPU gather corrupts V \
                     (V cosine {v_cos:.5} ≈ orthogonal). V-vs-inputK = {v_vs_k:.5} \
                     ({}).",
                    if v_vs_k.abs() > 0.9 {
                        "K/V SWAP: gathered V matches input K"
                    } else {
                        "not a clean K/V swap"
                    }
                );
                eprintln!("  re-run with FZ_TRACE=1 for per-sub-band byte offsets");
                panic!(
                    "V cosine {v_cos:.5} orthogonal while K cosine {k_cos:.5} survived — \
                     seal_to_chunk_images_cpu corrupts V (bug reproduced)"
                );
            } else {
                panic!(
                    "V cosine {v_cos:.5} is neither clean (>0.9) nor orthogonal (<=0.30) \
                     while K cosine {k_cos:.5} — inspect per-sub-band layout (FZ_TRACE=1)"
                );
            }
        }

        /// Same persist path as `persist_gather_preserves_v_vectors`, but the
        /// K arena is seeded as `Quantized(R16)` — the decode-time raw
        /// "F16 K + captured Q" carrier (128 B/sub-band) — while V stays F16
        /// (64 B/sub-band), the exact live-decode residency.
        ///
        /// This is a regression guard for the persist V-corruption bug: the
        /// `reassemble_block` CPU decoder used to read the quantized sub-bands
        /// as token-major and ignored the palette map, so V came back
        /// orthogonal (cosine ≈ 0) once the block was sealed to a quant format.
        /// The bands are dim-major and palette-routed; both K and V must now
        /// survive the seal → gather → dequant round-trip.
        #[test]
        fn persist_gather_preserves_v_vectors_r16_k() {
            let Some(device) = cuda_device_or_skip() else {
                eprintln!("no CUDA device — skipping persist_gather_preserves_v_vectors_r16_k");
                return;
            };
            let n_kv_head = FZ_N_KV_HEAD;
            let head_dim = FZ_HEAD_DIM;
            let n_tokens = 32usize;
            let chunk_size = candle_nn::CHUNK_SIZE;
            let sub_head_dim = head_dim / N_PALETTE;
            let per_layer = n_kv_head * head_dim * chunk_size;
            assert_eq!(n_tokens, chunk_size, "one full block");

            let policy = CompressionPolicy::new(1);

            // K arena = R16 (the decode-time carrier), V arena = F16.
            let backing = ChunkedKvBacking::new_with_format_adaptive(
                4,
                n_kv_head,
                head_dim,
                KvFormat::Quantized(QuantFormat::R16),
                KvFormat::Float(DType::F16),
                &device,
                FZ_ARENA_CAPACITY,
                Some(policy),
            )
            .expect("create chunked backing with R16 K + F16 V arenas");

            // ── Seed the R16-K decode residency EXACTLY ───────────────────
            // Build the raw dim-major R16 K bytes and the token-major F16 V
            // bytes and install them with `write_raw_sealed_chunk`, matching the
            // layout the decode kernel produces and the palette4 convert kernel
            // reads. `write_contiguous` would quantize K into R16 in the wrong
            // token-major layout — a test artifact — so it is deliberately not
            // used here.
            //
            // Orthogonal K/V patterns (different sinusoid frequencies) so a K/V
            // swap or a V-orthogonal corruption is visible in the cosines.
            let val_k = |h: usize, t: usize, d: usize| -> f32 {
                let i = (h * n_tokens + t) * head_dim + d;
                (i as f32 * 0.11).sin() + 0.3
            };
            let val_v = |h: usize, t: usize, d: usize| -> f32 {
                let i = (h * n_tokens + t) * head_dim + d;
                (i as f32 * 0.37).cos() - 0.2
            };

            // R16 K bytes, dim-major per (h, p) sub-band — sub_head_dim blocks of
            // 128 B; block[pd] = { F16 K[t=0..32] , u16 Q[t=0..32] }. Q is left
            // zero (the corruption is Q-value-independent). Matches
            // `dump_sequence_r16_kv_chunks` / the kernel's R16 src read.
            let mut k_bytes: Vec<u8> = Vec::new();
            for h in 0..n_kv_head {
                for p in 0..N_PALETTE {
                    for pd in 0..sub_head_dim {
                        let d = p * sub_head_dim + pd;
                        for t in 0..n_tokens {
                            let kh = f16::from_f32(val_k(h, t, d));
                            k_bytes.extend_from_slice(&kh.to_le_bytes());
                        }
                        k_bytes.extend_from_slice(&[0u8; 64]);
                    }
                }
            }
            // F16 V bytes, token-major per (h, p) sub-band: (token, dim).
            let mut v_bytes: Vec<u8> = Vec::new();
            for h in 0..n_kv_head {
                for p in 0..N_PALETTE {
                    for t in 0..n_tokens {
                        for pd in 0..sub_head_dim {
                            let d = p * sub_head_dim + pd;
                            let vh = f16::from_f32(val_v(h, t, d));
                            v_bytes.extend_from_slice(&vh.to_le_bytes());
                        }
                    }
                }
            }

            let slot = backing.alloc_sequence().unwrap();
            backing
                .write_raw_sealed_chunk(
                    slot,
                    0,
                    &k_bytes,
                    &v_bytes,
                    Arc::new(Vec::new()),
                    Arc::new(Vec::new()),
                    Arc::new(Vec::new()),
                    Arc::new(Vec::new()),
                )
                .expect("seed raw R16 K + F16 V chunk");
            backing.set_len(slot, n_tokens);
            let src = backing.record_turn(slot).unwrap();

            let idx_out = |h: usize, d: usize, t: usize| (h * head_dim + d) * chunk_size + t;
            let mut k_in = vec![0.0f32; per_layer];
            let mut v_in = vec![0.0f32; per_layer];
            for h in 0..n_kv_head {
                for d in 0..head_dim {
                    for t in 0..n_tokens {
                        k_in[idx_out(h, d, t)] = f16::from_f32(val_k(h, t, d)).to_f32();
                        v_in[idx_out(h, d, t)] = f16::from_f32(val_v(h, t, d)).to_f32();
                    }
                }
            }

            let copy_stream = cuda_stream(&device);
            let mut pinned: Option<PinnedBuf> = None;
            let warm = quantize_sealed_in_place(
                &backing,
                &[&src],
                &policy,
                &device,
                &copy_stream,
                &mut pinned,
            )
            .expect("quantize_sealed_in_place (R16 K)");
            assert_eq!(warm.len(), 1);
            let warm = &warm[0];

            let cpu_seq;
            let seq_ref: &SealedSequence = if warm.location == ArenaLocation::Cpu {
                warm
            } else {
                let migrated = backing
                    .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[warm])
                    .expect("migrate warm → CPU for the cold gather");
                cpu_seq = migrated.into_iter().next().unwrap();
                &cpu_seq
            };
            assert_eq!(seq_ref.location, ArenaLocation::Cpu);

            let images =
                seal_to_chunk_images(&backing, &device, seq_ref).expect("seal_to_chunk_images");
            assert!(!images.is_empty());
            let payload = &images[0].payload;

            let (k_rt, v_rt) = reassemble_block(
                payload,
                n_kv_head,
                head_dim,
                chunk_size,
                sub_head_dim,
                per_layer,
            )
            .expect("reassemble block 0");

            let k_cos = cosine(&k_rt, &k_in);
            let v_cos = cosine(&v_rt, &v_in);
            let v_vs_k = cosine(&v_rt, &k_in);
            eprintln!(
                "PERSIST-GATHER (R16 K)  K cosine = {k_cos:.5}   V cosine = {v_cos:.5}   \
                 (V-vs-inputK = {v_vs_k:.5})"
            );

            assert!(
                k_cos > 0.9,
                "K cosine {k_cos:.5} <= 0.9 — decoder/layout wrong before trusting V"
            );
            assert!(
                v_cos > 0.9,
                "V cosine {v_cos:.5} — V must survive the R16-K persist path \
                 (K cosine {k_cos:.5}, V-vs-inputK {v_vs_k:.5})"
            );
        }

        #[test]
        fn gather_then_scatter_round_trip_is_byte_identical() {
            let device = match candle::Device::cuda_if_available(0) {
                Ok(d @ candle::Device::Cuda(_)) => d,
                _ => return, // no GPU — skip
            };
            let dev = cuda_device(&device).unwrap();

            // Three scattered VRAM "chunks" with distinct byte patterns.
            let chunks: Vec<Vec<u8>> = vec![
                (0..320u32).map(|i| (i % 256) as u8).collect(),
                (0..512u32).map(|i| ((i * 5 + 7) % 256) as u8).collect(),
                (0..208u32).map(|i| ((i * 11 + 2) % 256) as u8).collect(),
            ];
            let src_gpus: Vec<_> = chunks.iter().map(|c| dev.memcpy_stod(c).unwrap()).collect();
            let src_ptrs: Vec<(i64, i64)> = {
                let stream = dev.cuda_stream();
                src_gpus
                    .iter()
                    .zip(&chunks)
                    .map(|(g, c)| (g.device_ptr(&stream).0 as i64, c.len() as i64))
                    .collect()
            };

            // gather → one contiguous host buffer.
            let gathered = gather_chunks(&device, &src_ptrs).unwrap();
            let concatenated: Vec<u8> = chunks.iter().flatten().copied().collect();
            assert_eq!(gathered, concatenated, "gather concatenates the chunks");

            // scatter the host buffer into a fresh set of VRAM chunks.
            let dst_gpus: Vec<_> = chunks
                .iter()
                .map(|c| unsafe { dev.alloc::<u8>(c.len()).unwrap() })
                .collect();
            let dst_ptrs: Vec<(i64, i64)> = {
                let stream = dev.cuda_stream();
                dst_gpus
                    .iter()
                    .zip(&chunks)
                    .map(|(g, c)| (g.device_ptr(&stream).0 as i64, c.len() as i64))
                    .collect()
            };
            scatter_chunks(&device, &dst_ptrs, &gathered).unwrap();

            for (i, c) in chunks.iter().enumerate() {
                let back = dev.memcpy_dtov(&dst_gpus[i]).unwrap();
                assert_eq!(&back, c, "chunk {i} round-trips byte-identical");
            }
        }

        /// The GPU golden path: gather scattered VRAM chunks and compute a
        /// Fletcher-32 per output chunk on the device, then check each against
        /// the CPU reference over the same bytes. Covers even/odd lengths.
        #[test]
        fn gather_with_goldens_matches_cpu_fletcher() {
            let device = match candle::Device::cuda_if_available(0) {
                Ok(d @ candle::Device::Cuda(_)) => d,
                _ => return, // no GPU — skip
            };
            let dev = cuda_device(&device).unwrap();

            let chunks: Vec<Vec<u8>> = vec![
                (0..320u32).map(|i| (i % 256) as u8).collect(),
                (0..513u32).map(|i| ((i * 5 + 7) % 256) as u8).collect(), // odd length
                (0..208u32).map(|i| ((i * 11 + 2) % 256) as u8).collect(),
            ];
            let src_gpus: Vec<_> = chunks.iter().map(|c| dev.memcpy_stod(c).unwrap()).collect();
            let src_ptrs: Vec<(i64, i64)> = {
                let stream = dev.cuda_stream();
                src_gpus
                    .iter()
                    .zip(&chunks)
                    .map(|(g, c)| (g.device_ptr(&stream).0 as i64, c.len() as i64))
                    .collect()
            };
            let split: Vec<usize> = chunks.iter().map(|c| c.len()).collect();

            let (blob, goldens) = gather_chunks_with_goldens(&device, &src_ptrs, &split).unwrap();
            let concatenated: Vec<u8> = chunks.iter().flatten().copied().collect();
            assert_eq!(blob, concatenated, "gather concatenates the chunks");
            assert_eq!(goldens.len(), chunks.len());
            for (i, c) in chunks.iter().enumerate() {
                assert_eq!(
                    goldens[i],
                    candle::fletcher::fletcher32(c),
                    "chunk {i} GPU golden != CPU reference"
                );
            }
        }

        #[test]
        fn scatter_rejects_a_size_mismatch() {
            let device = match candle::Device::cuda_if_available(0) {
                Ok(d @ candle::Device::Cuda(_)) => d,
                _ => return,
            };
            // A plan that wants 256 bytes, handed 100.
            let chunks = [(0x1000i64, 256i64)];
            assert!(scatter_chunks(&device, &chunks, &[0u8; 100]).is_err());
        }

        /// End-to-end byte-identity check for the cold-load bridge:
        /// `ColdLoadStager::pack` → `upload_async` (real pinned host →
        /// HtoD on the CUDA stream) → `kv_migrate` (scatter to
        /// VRAM-resident chunks) → `memcpy_dtov` (read back) →
        /// `assert_eq!` against the originals. If pinned alloc or the
        /// HtoD path is broken this fires.
        #[test]
        fn cold_load_stager_upload_and_scatter_round_trip_is_byte_identical() {
            use crate::persistence::cold_load::ColdLoadStager;
            use candle::cuda_backend::cudarc::driver::DevicePtr;
            use candle_nn::kv_cache::{kv_migrate, MigrationPlan};

            let device = match candle::Device::cuda_if_available(0) {
                Ok(d @ candle::Device::Cuda(_)) => d,
                _ => return, // no GPU — skip
            };
            let dev = cuda_device(&device).unwrap();

            // Three distinct-pattern "chunks" that get packed contiguously
            // through the stager and then scattered into three separately-
            // allocated VRAM regions — the exact shape of a cold load.
            let chunks: Vec<Vec<u8>> = vec![
                (0..512u32).map(|i| (i % 256) as u8).collect(),
                (0..256u32).map(|i| ((i * 7 + 13) % 256) as u8).collect(),
                (0..128u32).map(|i| ((i * 17 + 3) % 256) as u8).collect(),
            ];
            let total: usize = chunks.iter().map(|c| c.len()).sum();

            // Allocate destination VRAM chunks (the "arena chunks" the
            // cold-load path scatters into).
            let dst_gpus: Vec<_> = chunks
                .iter()
                .map(|c| unsafe { dev.alloc::<u8>(c.len()).unwrap() })
                .collect();
            let dst_ptrs: Vec<(i64, i64)> = {
                let stream = dev.cuda_stream();
                dst_gpus
                    .iter()
                    .zip(&chunks)
                    .map(|(g, c)| (g.device_ptr(&stream).0 as i64, c.len() as i64))
                    .collect()
            };

            // Pack → upload via the chunked-cold-load primitives.
            let mut stager = ColdLoadStager::with_preallocation(total);
            {
                let buf = stager.buffer_mut(total);
                let mut off = 0;
                for c in &chunks {
                    buf[off..off + c.len()].copy_from_slice(c);
                    off += c.len();
                }
            }
            let stream = dev.cuda_stream();
            let staging = stager
                .upload_async(dev, &stream, total)
                .expect("upload_async to device");

            // Scatter the contiguous staging blob into the dst chunks.
            let mut plan = MigrationPlan::new();
            let mut offset = 0i64;
            for &(ptr, len) in &dst_ptrs {
                plan.push(staging.base_ptr + offset, ptr, len);
                offset += len;
            }
            kv_migrate(&device, &plan).expect("kv_migrate scatter");
            drop(staging);

            for (i, c) in chunks.iter().enumerate() {
                let back = dev.memcpy_dtov(&dst_gpus[i]).unwrap();
                assert_eq!(
                    &back, c,
                    "chunk {i} bytes survive the pinned-host → HtoD → scatter chain"
                );
            }
        }
    }
}

pub use cuda_impl::{
    gather_chunks, load_stream_into_hot, load_turn_into_hot, scatter_chunks, seal_to_chunk_images,
};
