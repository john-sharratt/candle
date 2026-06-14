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
        kv_migrate, ArenaLocation, ChunkedKvBacking, MigrationPlan, SealedSequence,
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

        let ptrs = backing.resolve_sealed_chunk_ptrs(seq)?;
        let blob = gather_chunks(device, &ptrs)?;
        let mut cursor = 0usize;
        let mut images = Vec::with_capacity(seq.chunks.len());
        for sc in &seq.chunks {
            let n = sc.byte_size as usize;
            if cursor + n > blob.len() {
                return Err(candle::Error::Msg(format!(
                    "seal_to_chunk_images: blob underrun (need {n} at {cursor}, have {})",
                    blob.len()
                )));
            }
            let kv_bytes = blob[cursor..cursor + n].to_vec();
            cursor += n;
            let (k_formats, v_formats) = backing.kv_formats_for_gids(&sc.gids)?;
            images.push(ChunkImage {
                token_count: sc.token_count,
                payload: ChunkPayload {
                    offset: sc.offset,
                    k_formats: k_formats.iter().map(|f| f.to_tag()).collect(),
                    v_formats: v_formats.iter().map(|f| f.to_tag()).collect(),
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
        use candle_nn::kv_cache::arena_gid_stride;

        use crate::persistence::record::ChunkPayload;

        let arena_chunks = arena_gid_stride();
        let arena_info = backing.resolve_arena_info()?;
        // Pull each unique (arena, chunk) slot's bytes into a single
        // blob, then split per `sc.byte_size` like the GPU path. This
        // preserves the SealedChunk's per-chunk byte_size accounting
        // exactly, including the per-(h, p) GID dedup that
        // `arena_byte_size` already computed.
        let mut seen = std::collections::HashSet::new();
        let mut blob: Vec<u8> = Vec::new();
        for chunk in &seq.chunks {
            for gid in chunk.gids.as_slice() {
                let raw = gid.raw();
                if !seen.insert(raw) {
                    continue;
                }
                let arena_idx = (raw as usize) / arena_chunks;
                let chunk_idx = (raw as usize) % arena_chunks;
                let info = arena_info.get(arena_idx).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "seal_to_chunk_images_cpu: arena_idx {arena_idx} out of range"
                    ))
                })?;
                let stride = info.chunk_byte_stride as usize;
                let start = blob.len();
                blob.resize(start + stride, 0);
                backing.read_chunk_into_bytes(
                    arena_idx,
                    chunk_idx,
                    &mut blob[start..start + stride],
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
            let (k_formats, v_formats) = backing.kv_formats_for_gids(&sc.gids)?;
            images.push(ChunkImage {
                token_count: sc.token_count,
                payload: ChunkPayload {
                    offset: sc.offset,
                    k_formats: k_formats.iter().map(|f| f.to_tag()).collect(),
                    v_formats: v_formats.iter().map(|f| f.to_tag()).collect(),
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
        // layer (matches the legacy `recover_turn` + `load_to_hot`
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

        for batch in &plan.chunks {
            let stats = crate::persistence::pipeline::run_pipeline(
                persistence,
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
