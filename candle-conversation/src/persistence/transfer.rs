//! VRAM↔RAM transfer orchestration (§3, §4, §16.6 of
//! `docs/kv_tier_migration.md`).
//!
//! `transfer.rs` drives `candle-nn`'s `kv_migrate` scatter/gather kernel:
//!
//! - [`gather_chunks`] — resolve a set of scattered VRAM chunks into one
//!   contiguous host buffer (the core of eviction).
//! - [`scatter_chunks`] — write a contiguous host buffer back into a set of
//!   VRAM chunks (the core of a load).
//! - [`evict_to_warm`] / [`load_to_hot`] — the `SealedSequence`-level evict
//!   and load paths, thin glue over the two primitives plus the
//!   [`WarmPool`](super::warm_pool::WarmPool).
//!
//! `candle-conversation` is CUDA-only, so this module is unconditionally
//! compiled — the gather/scatter helpers and `kv_migrate` plan it builds
//! all assume a CUDA device.

mod cuda_impl {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::{Device, Result};
    use candle_nn::kv_cache::{
        kv_migrate, ChunkedKvBacking, KvFormat, MigrationPlan, SealedSequence,
    };

    use crate::persistence::resume::ChunkImage;
    use crate::persistence::streams::StreamId;
    use crate::persistence::warm_pool::WarmPool;

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
    /// seal-time redo-log write — the representation `load_stream` can
    /// rebuild VRAM chunks from.
    pub fn seal_to_chunk_images(
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
            let (k_formats, v_formats) = backing.sealed_chunk_kv_formats(sc)?;
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

    /// Evict a sealed sequence VRAM→RAM: gather its chunks into the warm
    /// pool as [`ChunkImage`]s, marked dirty (not yet durable on disk).
    pub fn evict_to_warm(
        backing: &ChunkedKvBacking,
        seq: &SealedSequence,
        device: &Device,
        stream_id: StreamId,
        warm_pool: &mut WarmPool,
    ) -> Result<()> {
        let images = seal_to_chunk_images(backing, device, seq)?;
        warm_pool.insert(stream_id, images, true);
        Ok(())
    }

    /// Load a stream's warm chunks back into VRAM — the warm→hot reload.
    ///
    /// Built on [`load_stream`] / the allocation keystone, so it rebuilds a
    /// fresh `SealedSequence` (new GIDs) rather than scattering into stale,
    /// freed chunks. This is what makes a warm eviction that *frees* VRAM
    /// reloadable. Threads the caller's reusable
    /// [`ColdLoadStager`](crate::persistence::cold_load::ColdLoadStager)
    /// through so the warm→hot HtoD goes through pinned host memory, not
    /// the driver's pageable bounce.
    pub fn load_to_hot(
        backing: &ChunkedKvBacking,
        device: &Device,
        warm_images: &[ChunkImage],
        stager: &mut crate::persistence::cold_load::ColdLoadStager,
    ) -> Result<SealedSequence> {
        load_stream(backing, device, warm_images, stager)
    }

    /// Reconstruct one layer's [`SealedSequence`] from its recovered
    /// [`ChunkImage`]s — the body of the design's `load_stream` (§16.12).
    ///
    /// Built entirely on the allocation keystone and the layout-agnostic
    /// migration path: a scratch slot is allocated, `alloc_sealed_block`
    /// materialises each chunk's GIDs in arenas of its persisted `KvFormat`,
    /// `set_block_window` stamps the `offset` / `token_count`, `record_turn`
    /// snapshots a `SealedSequence` with real GIDs, and `scatter_chunks`
    /// fills the opaque `kv_bytes` through `resolve_sealed_chunk_ptrs`. The
    /// scratch slot is then freed — the returned sequence keeps the chunks
    /// alive via its `ChunkGid` `Arc`s.
    ///
    /// Gather and scatter route through the *same* `resolve_sealed_chunk_ptrs`
    /// and the *same* `chunk_byte_stride`-from-`format`; there is no second
    /// layout calculation, so the round trip is correct by construction.
    pub fn load_stream(
        backing: &ChunkedKvBacking,
        device: &Device,
        chunks: &[ChunkImage],
        stager: &mut crate::persistence::cold_load::ColdLoadStager,
    ) -> Result<SealedSequence> {
        use std::sync::Arc;

        let dev = cuda_device(device)?;
        let slot = backing.alloc_sequence()?;
        let decode_formats = |tags: &[u8], side: &str| -> Result<Vec<KvFormat>> {
            tags.iter()
                .map(|&t| {
                    KvFormat::from_tag(t).ok_or_else(|| {
                        candle::Error::Msg(format!("load_stream: unrecognised {side} format tag {t}"))
                    })
                })
                .collect()
        };
        let mut build = || -> Result<SealedSequence> {
            let mut total_tokens = 0usize;
            for (block_idx, image) in chunks.iter().enumerate() {
                let k_formats = decode_formats(&image.payload.k_formats, "k")?;
                let v_formats = decode_formats(&image.payload.v_formats, "v")?;
                backing.alloc_sealed_block(
                    slot,
                    block_idx,
                    &k_formats,
                    &v_formats,
                    Arc::new(image.payload.k_pal.clone()),
                    Arc::new(image.payload.v_pal.clone()),
                    Arc::new(image.payload.k_scale.clone()),
                    Arc::new(image.payload.v_scale.clone()),
                )?;
                backing.set_block_window(
                    slot,
                    block_idx,
                    image.payload.offset,
                    image.token_count as u32,
                )?;
                total_tokens += image.token_count as usize;
            }
            let seq = backing.record_turn(slot, total_tokens)?;
            let ptrs = backing.resolve_sealed_chunk_ptrs(&seq)?;

            // Bridge path: pack the per-chunk kv_bytes into the
            // reusable pinned host scratch and HtoD once into a fresh
            // device staging slice; then `kv_migrate` scatters into the
            // freshly-allocated VRAM chunks. The bytes source for the
            // HtoD is real pinned memory, so the driver does not need
            // to synthesise a pageable bounce buffer. See
            // `crate::persistence::cold_load` for the bridge-vs-GDS
            // rationale.
            let total_bytes: usize = chunks.iter().map(|c| c.payload.kv_bytes.len()).sum();
            if total_bytes == 0 {
                return Ok(seq);
            }
            let _packed_len = {
                let packed = stager.pack(chunks.iter().map(|c| c.payload.kv_bytes.as_slice()))?;
                packed.len()
            };
            let stream = dev.cuda_stream();
            let staging = stager.upload_async(dev, &stream, total_bytes)?;

            let mut plan = MigrationPlan::new();
            let mut offset = 0i64;
            for &(ptr, len) in &ptrs {
                plan.push(staging.base_ptr + offset, ptr, len);
                offset += len;
            }
            kv_migrate(device, &plan)?;
            // `staging.slice` drops here, freeing the device allocation
            // after `kv_migrate` returns.
            drop(staging);
            Ok(seq)
        };
        let result = build();
        // The scratch slot is no longer needed — the returned sequence owns
        // the chunks through its `ChunkGid` Arcs. Free even on the error path.
        backing.free_sequence(slot)?;
        result
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

            // Pack → upload via the bridge primitive.
            let mut stager = ColdLoadStager::new();
            {
                let packed = stager
                    .pack(chunks.iter().map(|c| c.as_slice()))
                    .expect("pack into pinned scratch");
                assert_eq!(packed.len(), total);
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
    evict_to_warm, gather_chunks, load_stream, load_to_hot, scatter_chunks, seal_to_chunk_images,
};
