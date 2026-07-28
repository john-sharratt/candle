//! Chunk migration and format conversion operations.
//!
//! This module provides operations for moving and converting KV cache chunks
//! between different storage formats (float/quantized) and locations (GPU/CPU):
//!
//! - [`migrate_chunk`] - Move a chunk to a different arena type
//! - [`copy_chunk_data`] - Copy data between arenas of the same type
//! - [`convert_chunk_data`] - Convert data between different formats
//! - [`prepare`] - Prepare chunks for kernel execution
//! - [`reconcile`] - Migrate chunks to match a storage policy (batched GPU quantization)

// Import from parent module (chunked/mod.rs)

use super::gid_pool::ChunkGid;
use super::head_gids::{HeadGids, GIDS_PER_HEAD};
use super::types::{SealedChunk, SealedSequence};
use super::{arena_gid_stride, CHUNK_SIZE};
use super::{Arena, ArenaKey, ChunkedKvBacking};
// Import from kv_cache module (grandparent)
use crate::kv_cache::arena_table::{ArenaLocation, N_PALETTE};
use crate::kv_cache::KvFormat;
#[cfg(feature = "cuda")]
use crate::kv_cache::QuantFormat;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::SELECT_FMT_F16;
use candle::quantized::QTensor;
use candle::{DType, Device, Result, Tensor};

// Shared production thresholds are defined in compression_policy.rs and are
// consumed here directly so the runtime and table harness stay in sync.

/// Upper bound on one hot→warm migration batch's contiguous staging buffer
/// (host pinned/heap AND transient GPU staging). Kept deliberately small: the
/// warm tier fills host RAM with thousands of 16 MiB CPU arenas, so a *large*
/// contiguous host allocation (the old 2 GiB) can fail on a fragmented heap and
/// abort the process. At 512 MiB a full ~1.4 GiB pass migrates in ~3 batches
/// (≈3 DtoH syncs) — still far below the per-layer 48 that made `copy_ms` the
/// drain bottleneck, while bounding every staging allocation to a size the heap
/// can reliably supply. The persistence thread pre-sizes its reusable scratch to
/// this at startup (clean heap), so it never has to re-grow under fragmentation.
pub const MIGRATION_STAGING_CAP_BYTES: usize = 512 * 1024 * 1024;

/// Source formats the quantize-on-evict kernel can ingest directly:
/// GPU `Float` (any dtype) or `Quantized(R16)`. Anything else has to
/// take the format-preserving migration path because the selection /
/// conversion kernels are only compiled for these source layouts.
#[cfg(feature = "cuda")]
#[inline]
pub(super) fn needs_reconcile_source_format(format: KvFormat) -> bool {
    matches!(
        format,
        KvFormat::Float(_) | KvFormat::Quantized(QuantFormat::R16)
    )
}

/// Byte size of one chunk slot in an arena — the unit
/// [`super::migrate::kv_migrate_on`] uses for its DMA stride, and the
/// chunk_byte_stride [`super::backing::BackingInner::resolve_arena_info`]
/// reports.
///
/// For Float arenas: `prod(per_chunk_dims) × dtype.size_in_bytes`.
/// For Quantized arenas: `(elems_per_chunk / block_size) × type_size`,
/// inferred from the arena's total element count divided by its
/// arena_chunks (so it handles any head_dim, not just the canonical
/// 128).
///
/// Used by the cuda-gated `migrate_sealed_to_gpu_batch_async` path and
/// its test fixtures; cfg'd to that feature so non-cuda builds don't
/// see it as dead code.
#[cfg(feature = "cuda")]
fn chunk_byte_size_of(arena: &Arena) -> Result<usize> {
    match arena {
        Arena::Float { data, dtype, .. } => {
            let dims = data.dims();
            if dims.is_empty() {
                return Err(candle::Error::Msg(
                    "chunk_byte_size_of: arena tensor has zero dims".into(),
                ));
            }
            let n_elems: usize = dims[1..].iter().product();
            Ok(n_elems * dtype.size_in_bytes())
        }
        Arena::Quantized { format, data, .. } => {
            let kv_fmt = crate::kv_cache::KvFormat::Quantized(*format);
            let arena_chunks = super::arena_chunks_for_format(kv_fmt);
            let total_elems = data.shape().elem_count();
            if arena_chunks == 0 || total_elems == 0 {
                return Err(candle::Error::Msg(
                    "chunk_byte_size_of: quantized arena has zero elements or chunks".into(),
                ));
            }
            let elems_per_chunk = total_elems / arena_chunks;
            let ggml = format.to_ggml_dtype();
            Ok((elems_per_chunk / ggml.block_size()) * ggml.type_size())
        }
    }
}

/// Read one chunk-slot's bytes from a CPU `Arena::Float` at
/// `(arena, chunk_idx)` into `dst`. The reverse of
/// [`ChunkedKvBacking::write_chunk_from_pinned_bytes`]; same dtype
/// dispatch (`bf16` / `f16` / `f32`).
fn read_chunk_into_pinned_bytes(arena: &Arena, chunk_idx: usize, dst: &mut [u8]) -> Result<()> {
    match arena {
        Arena::Float { data, dtype, .. } => {
            let dims = data.dims();
            if dims.is_empty() {
                return Err(candle::Error::Msg(
                    "read_chunk_into_pinned_bytes: arena tensor has zero dims".into(),
                ));
            }
            let n_elems: usize = dims[1..].iter().product();
            let expected_bytes = n_elems * dtype.size_in_bytes();
            if dst.len() != expected_bytes {
                return Err(candle::Error::Msg(format!(
                    "read_chunk_into_pinned_bytes: expected {expected_bytes} bytes, got {}",
                    dst.len()
                )));
            }
            // Narrow → flatten → to_vec is the only stable path to a
            // contiguous byte read of a CPU Tensor in candle today. The
            // Vec allocation is one extra copy on top of the pinned
            // memcpy; acceptable for the warm→hot path (rare, eviction-
            // driven). A direct `&[u8]` view into CpuStorage would need
            // a candle-core API change.
            let view = data.narrow(0, chunk_idx, 1)?;
            let flat = view.flatten_all()?;
            match dtype {
                DType::BF16 => {
                    let v: Vec<half::bf16> = flat.to_vec1::<half::bf16>()?;
                    // SAFETY: bf16 is 2-byte POD; v.len() == n_elems
                    // (validated via expected_bytes above).
                    let bytes = unsafe {
                        std::slice::from_raw_parts(v.as_ptr() as *const u8, expected_bytes)
                    };
                    dst.copy_from_slice(bytes);
                }
                DType::F16 => {
                    let v: Vec<half::f16> = flat.to_vec1::<half::f16>()?;
                    let bytes = unsafe {
                        std::slice::from_raw_parts(v.as_ptr() as *const u8, expected_bytes)
                    };
                    dst.copy_from_slice(bytes);
                }
                DType::F32 => {
                    let v: Vec<f32> = flat.to_vec1::<f32>()?;
                    let bytes = unsafe {
                        std::slice::from_raw_parts(v.as_ptr() as *const u8, expected_bytes)
                    };
                    dst.copy_from_slice(bytes);
                }
                other => {
                    return Err(candle::Error::Msg(format!(
                        "read_chunk_into_pinned_bytes: unsupported Float dtype {other:?}"
                    )))
                }
            }
            Ok(())
        }
        Arena::Quantized { data, .. } => {
            // Same flat-slab story as `write_chunk_from_pinned_bytes`:
            // one `chunk_byte_stride` of bytes starting at
            // `chunk_idx * dst.len()`. The HtoD scatter on the next
            // phase rebuilds the GPU arena chunk-for-chunk from this
            // pinned scratch.
            let byte_offset = chunk_idx * dst.len();
            data.read_bytes_at(byte_offset, dst)?;
            Ok(())
        }
    }
}

/// Per-block parameter bundle for [`ChunkedKvBacking::alloc_sealed_blocks_bulk`].
///
/// One spec carries everything that one call to
/// [`ChunkedKvBacking::alloc_sealed_block`] +
/// [`ChunkedKvBacking::set_block_window`] would need. The cold-load
/// path constructs a `Vec<BlockAllocSpec>` for an entire turn's worth
/// of blocks (per layer) and submits them in one bulk call, collapsing
/// the per-block `state.write()` lock churn (3 acquisitions/block ×
/// thousands of blocks) into 1 acquisition/layer.
#[derive(Clone)]
pub struct BlockAllocSpec {
    pub block_idx: usize,
    pub k_formats: Vec<KvFormat>,
    pub v_formats: Vec<KvFormat>,
    pub k_pal: std::sync::Arc<Vec<u8>>,
    pub v_pal: std::sync::Arc<Vec<u8>>,
    pub k_scale: std::sync::Arc<Vec<f32>>,
    pub v_scale: std::sync::Arc<Vec<f32>>,
    pub offset: u16,
    pub usage: u32,
}

impl ChunkedKvBacking {
    /// Read the raw bytes of one chunk slot into `dst`. Works for both
    /// `Arena::Float` (any supported dtype) and `Arena::Quantized`
    /// (CPU or GPU). The destination buffer must be exactly the
    /// arena's `chunk_byte_stride` bytes — call sites use
    /// [`crate::kv_cache::arena_table::ResolvedArenaInfo::chunk_byte_stride`]
    /// to size it.
    pub fn read_chunk_into_bytes(
        &self,
        arena_idx: usize,
        chunk_idx: usize,
        dst: &mut [u8],
    ) -> candle::Result<()> {
        self.inner.storage.read(|s| {
            let arena = s.arenas().get(&arena_idx).ok_or_else(|| {
                candle::Error::Msg(format!("read_chunk_into_bytes: arena {arena_idx} missing"))
            })?;
            read_chunk_into_pinned_bytes(arena, chunk_idx, dst)
        })?
    }

    /// Migrate a chunk from one arena type to another.
    ///
    /// This is the central operation for moving data between formats (float/quant)
    /// and locations (GPU/CPU). The source chunk data is copied/converted to a
    /// newly allocated chunk in the target arena type.
    ///
    /// # Arguments
    /// * `source_gid` - Global chunk ID of the source chunk
    /// * `target_key` - The format/location to migrate to
    ///
    /// # Returns
    /// The global chunk ID of the newly allocated chunk in the target arena type.
    pub fn migrate_chunk(&self, source_gid: i64, target_key: ArenaKey) -> Result<ChunkGid> {
        if source_gid < 0 {
            candle::bail!("migrate_chunk: invalid source chunk ID {}", source_gid);
        }
        let source_gid = source_gid as usize;

        let arena_chunks = arena_gid_stride();
        let source_arena_idx = source_gid / arena_chunks;
        let source_chunk_idx = source_gid % arena_chunks;

        // Capture params for closure
        let n_kv_head = self.inner.n_kv_head;
        let chunk_size = CHUNK_SIZE;
        let head_dim = self.inner.head_dim;

        // Allocate destination GID through the pool ? this is the single allocation path.
        let new_gid = {
            let _state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            self.alloc_chunk_for_key(target_key.clone())?
        };

        // Perform the data copy/conversion inside a storage write lock.
        let arena_idx = new_gid.arena_idx();
        let chunk_idx = new_gid.chunk_idx();
        let copy_result = self.inner.storage.try_write(|s| {
            // Re-validate the source by MAP LOOKUP, not an index-vs-len bound:
            // storage `arenas` is a keyed map with holes once any arena has been
            // tombstoned, so `arena_count()` (the map LEN) undercounts and a
            // perfectly valid high-index source would be misjudged "tombstoned".
            // That misjudgement made this path skip the copy and hand back a
            // detached pseudo-gid aliasing the live source (see below) — the
            // stale-gid corruption at the root of the CFW crash/drift.
            let Some(source_key) = s.arena_key(source_arena_idx) else {
                return Ok(false);
            };

            if source_key == target_key {
                Self::copy_chunk_data_static(
                    s.arenas_mut(),
                    source_arena_idx,
                    source_chunk_idx,
                    arena_idx,
                    chunk_idx,
                    n_kv_head,
                    chunk_size,
                    head_dim,
                )?;
            } else {
                Self::convert_chunk_data_static(
                    s.arenas_mut(),
                    source_arena_idx,
                    source_chunk_idx,
                    source_key,
                    arena_idx,
                    chunk_idx,
                    target_key,
                    n_kv_head,
                    chunk_size,
                    head_dim,
                )?;
            }
            Ok(true)
        });

        match copy_result {
            Ok(true) => Ok(new_gid),
            Ok(false) => {
                // The source arena is genuinely absent from storage. The caller
                // holds the source gid's Arc across this call, so a live source
                // cannot be tombstoned under us (`try_tombstone` requires
                // `live == 0`) — absence here means an arena was freed while
                // chunks still referenced it: the exact invariant violation
                // behind the stale-gid KV corruption. Fail loudly. The old
                // fallback returned `ChunkGid::detached(source_gid)` — an
                // UNTRACKED alias of the freed id that let its arena index be
                // re-tenanted under a live chunk window (illegal-address when
                // strides differ, silent cross-context KV reads when not).
                // new_gid drops here, returning the dest GID to the pool.
                Err(candle::Error::Msg(format!(
                    "migrate_chunk: source arena {source_arena_idx} (gid {source_gid}) vanished \
                     while its chunk is still referenced — arena freed with live KV"
                )))
            }
            Err(e) => {
                // Data copy failed ? new_gid drops here, returning GID to pool.
                Err(e)
            }
        }
    }

    /// Static helper to copy chunk data (no &self reference for use in closures).
    #[allow(clippy::too_many_arguments)]
    fn copy_chunk_data_static(
        arenas: &mut ahash::AHashMap<usize, Arena>,
        src_arena: usize,
        src_chunk: usize,
        dst_arena: usize,
        dst_chunk: usize,
        _n_kv_head: usize,
        chunk_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        if src_arena == dst_arena {
            // Same-arena copy: clone/copy source data first to release the immutable
            // borrow, then take a mutable borrow for the write.
            let sub_head_dim = (head_dim / N_PALETTE).max(1);
            let elems_per_chunk = chunk_size * sub_head_dim;
            match arenas.get(&src_arena) {
                Some(Arena::Float { data, .. }) => {
                    let chunk = data.narrow(0, src_chunk, 1)?.copy()?;
                    match arenas.get_mut(&dst_arena) {
                        Some(Arena::Float { data, .. }) => data.slice_set(&chunk, 0, dst_chunk)?,
                        _ => candle::bail!("copy_chunk_data: same-arena Float mismatch"),
                    }
                }
                Some(Arena::Quantized { data, .. }) => {
                    let src_data_clone = data.clone();
                    let src_elem_offset = src_chunk * elems_per_chunk;
                    let dst_elem_offset = dst_chunk * elems_per_chunk;
                    match arenas.get_mut(&dst_arena) {
                        Some(Arena::Quantized { data: dst_data, .. }) => {
                            dst_data.slice_range_copy(
                                &src_data_clone,
                                src_elem_offset,
                                dst_elem_offset,
                                elems_per_chunk,
                            )?;
                        }
                        _ => candle::bail!("copy_chunk_data: same-arena Quantized mismatch"),
                    }
                }
                None => candle::bail!("copy_chunk_data: src arena {} not found", src_arena),
            }
            return Ok(());
        }

        // Different arenas: extract source data first (clone to release immutable borrow),
        // then mutably access destination.
        match arenas.get(&src_arena) {
            Some(Arena::Float { data, .. }) => {
                let chunk = data.narrow(0, src_chunk, 1)?.copy()?;
                match arenas.get_mut(&dst_arena) {
                    Some(Arena::Float { data: dst_data, .. }) => {
                        dst_data.slice_set(&chunk, 0, dst_chunk)?;
                        Ok(())
                    }
                    _ => candle::bail!("copy_chunk_data: mismatched arena types"),
                }
            }
            Some(Arena::Quantized { data: src_data, .. }) => {
                let sub_head_dim = (head_dim / N_PALETTE).max(1);
                let elems_per_chunk = chunk_size * sub_head_dim;
                let src_elem_offset = src_chunk * elems_per_chunk;
                let dst_elem_offset = dst_chunk * elems_per_chunk;
                let src_dtype = src_data.dtype();
                // Clone source data to release the immutable borrow before mutably borrowing dst.
                let src_data_clone = src_data.clone();
                match arenas.get_mut(&dst_arena) {
                    Some(Arena::Quantized { data: dst_data, .. }) => {
                        if src_dtype == dst_data.dtype() {
                            // Same-format copy: direct byte-level memcpy, no dequant/requant
                            dst_data.slice_range_copy(
                                &src_data_clone,
                                src_elem_offset,
                                dst_elem_offset,
                                elems_per_chunk,
                            )?;
                            Ok(())
                        } else {
                            candle::bail!(
                                "copy_chunk_data: quant dtype mismatch ({:?} vs {:?}) - K/V target key routing error",
                                src_dtype,
                                dst_data.dtype()
                            )
                        }
                    }
                    _ => candle::bail!("copy_chunk_data: mismatched arena types"),
                }
            }
            None => candle::bail!("copy_chunk_data: src arena {} not found", src_arena),
        }
    }

    /// Static helper to convert chunk data between different arena types.
    #[allow(clippy::too_many_arguments)]
    fn convert_chunk_data_static(
        arenas: &mut ahash::AHashMap<usize, Arena>,
        src_arena: usize,
        src_chunk: usize,
        src_key: ArenaKey,
        dst_arena: usize,
        dst_chunk: usize,
        dst_key: ArenaKey,
        _n_kv_head: usize,
        chunk_size: usize,
        head_dim: usize,
    ) -> Result<()> {
        use ArenaLocation::*;

        // Flat arena: each chunk = chunk_size * sub_head_dim (one head, one palette sub-band).
        // Arenas are allocated with shape (arena_chunks, CHUNK_SIZE, sub_head_dim) where
        // sub_head_dim = head_dim / N_PALETTE.  Using head_dim here produces a 4× offset
        // overrun on typical models (head_dim=128, N_PALETTE=4 → sub_head_dim=32).
        let sub_head_dim = (head_dim / N_PALETTE).max(1);
        let elems_per_chunk = chunk_size * sub_head_dim;

        // Helper to get device from an arena
        let get_device = |arena: &Arena| -> Device {
            match arena {
                Arena::Float { data, .. } => data.device().clone(),
                Arena::Quantized { data, .. } => data.device().clone(),
            }
        };

        let _src_device = arenas
            .get(&src_arena)
            .map(get_device)
            .unwrap_or(Device::Cpu);
        let dst_device = arenas
            .get(&dst_arena)
            .map(get_device)
            .unwrap_or(Device::Cpu);

        match (
            src_key.location,
            &src_key.format,
            dst_key.location,
            &dst_key.format,
        ) {
            // GPU Float → GPU Quant (quantization)
            // IMPORTANT: Kernel expects token-oriented layout for quantized data:
            //   Each 32-element block contains 32 tokens for a single dimension.
            //   Float layout: [token, dim] — flat arena, one head one side
            //   Quant layout: [dim, token] — token-oriented for quantized
            //
            // For Q4_0/Q8_0 with chunk_size=32, we use a fused transpose+quantize kernel
            // that reads from [T, D] and writes to [D, T] quantized layout in one pass.
            #[cfg(feature = "cuda")]
            (Gpu, KvFormat::Float(_), Gpu, KvFormat::Quantized(_)) => {
                // Extract src data first to release immutable borrow before mutably borrowing dst.
                let chunk_3d = {
                    let src_data = arenas
                        .get(&src_arena)
                        .ok_or_else(|| {
                            candle::Error::Msg(format!("src arena {src_arena} not found"))
                        })?
                        .float_data()?;
                    // Flat arena: shape (arena_chunks, chunk_size, head_dim)
                    let chunk_float = src_data.narrow(0, src_chunk, 1)?.squeeze(0)?; // (chunk_size, head_dim)
                    chunk_float.unsqueeze(0)? // (1, chunk_size, head_dim) for kernel
                };

                let dst_elem_offset = dst_chunk * elems_per_chunk;
                let dst_data = arenas
                    .get_mut(&dst_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("dst arena {dst_arena} not found")))?
                    .quantized_data_mut()?;

                fn can_fuse(fmt: &KvFormat, cs: usize) -> bool {
                    matches!(
                        fmt,
                        KvFormat::Quantized(
                            crate::kv_cache::QuantFormat::Q4_0
                                | crate::kv_cache::QuantFormat::Q4_1
                                | crate::kv_cache::QuantFormat::Q8_0
                                | crate::kv_cache::QuantFormat::Q8_1
                        )
                    ) && cs == 32
                }

                if can_fuse(&dst_key.format, chunk_size) {
                    dst_data.quantize_transposed_into(&chunk_3d, dst_elem_offset)?;
                } else {
                    let transposed = chunk_3d.transpose(1, 2)?.contiguous()?;
                    dst_data.quantize_into(&transposed, dst_elem_offset)?;
                }
                Ok(())
            }

            #[cfg(not(feature = "cuda"))]
            (Gpu, KvFormat::Float(_), Gpu, KvFormat::Quantized(_)) => {
                candle::bail!("GPU quantization requires CUDA feature")
            }

            // GPU Quant → GPU Float (dequantization) - efficient per-chunk conversion
            // IMPORTANT: Quantized data is in token-oriented layout [head, dim, token]
            // but float arenas use channel-oriented layout [head, token, dim].
            // We dequantize to a temp buffer and transpose back.
            #[cfg(feature = "cuda")]
            (Gpu, KvFormat::Quantized(_), Gpu, KvFormat::Float(dtype)) => {
                // Clone src QTensor first to avoid borrow conflicts.
                let src_data_clone = arenas
                    .get(&src_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("src arena {src_arena} not found")))?
                    .quantized_data()?
                    .clone();

                // Flat arena: each chunk = elems_per_chunk
                let src_elem_offset = src_chunk * elems_per_chunk;

                let device = arenas
                    .get(&dst_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("dst arena {dst_arena} not found")))?
                    .float_data()?
                    .device()
                    .clone();

                // Temp buffer for token-oriented layout: (1, sub_head_dim, chunk_size)
                let temp_shape = (1, sub_head_dim, chunk_size);
                let mut temp = Tensor::zeros(temp_shape, DType::F16, &device)?;

                src_data_clone.dequantize_into(&mut temp, src_elem_offset, 0, elems_per_chunk)?;

                // Transpose from [1, D/P, T] to [1, T, D/P] = (1, chunk_size, sub_head_dim)
                let transposed = temp.transpose(1, 2)?.contiguous()?;

                let final_data = if *dtype != DType::F16 {
                    transposed.to_dtype(*dtype)?
                } else {
                    transposed
                };

                // slice_set into flat arena at (dst_chunk, :, :)
                arenas
                    .get_mut(&dst_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("dst arena {dst_arena} not found")))?
                    .float_data_mut()?
                    .slice_set(&final_data, 0, dst_chunk)?;

                Ok(())
            }

            #[cfg(not(feature = "cuda"))]
            (Gpu, KvFormat::Quantized(_), Gpu, KvFormat::Float(_)) => {
                candle::bail!("GPU dequantization requires CUDA feature")
            }

            // GPU Float → CPU Float (D2H copy)
            (Gpu, KvFormat::Float(_), Cpu, KvFormat::Float(_)) => {
                let chunk = arenas
                    .get(&src_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("src arena {src_arena} not found")))?
                    .float_data()?
                    .narrow(0, src_chunk, 1)?
                    .to_device(&Device::Cpu)?;
                arenas
                    .get_mut(&dst_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("dst arena {dst_arena} not found")))?
                    .float_data_mut()?
                    .slice_set(&chunk, 0, dst_chunk)?;
                Ok(())
            }

            // CPU Float → GPU Float (H2D copy)
            (Cpu, KvFormat::Float(_), Gpu, KvFormat::Float(_)) => {
                let chunk = arenas
                    .get(&src_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("src arena {src_arena} not found")))?
                    .float_data()?
                    .narrow(0, src_chunk, 1)?
                    .to_device(&dst_device)?;
                arenas
                    .get_mut(&dst_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("dst arena {dst_arena} not found")))?
                    .float_data_mut()?
                    .slice_set(&chunk, 0, dst_chunk)?;
                Ok(())
            }

            // GPU Float → CPU Quant
            (Gpu, KvFormat::Float(_), Cpu, KvFormat::Quantized(_)) => {
                let chunk_float = {
                    let src_data = arenas
                        .get(&src_arena)
                        .ok_or_else(|| {
                            candle::Error::Msg(format!("src arena {src_arena} not found"))
                        })?
                        .float_data()?;
                    // Flat arena: chunk = (chunk_size, head_dim)
                    src_data
                        .narrow(0, src_chunk, 1)?
                        .squeeze(0)?
                        .to_device(&Device::Cpu)?
                        .contiguous()?
                };

                let ggml = dst_key.format.as_quant().unwrap().to_ggml_dtype();
                let chunk_quant = QTensor::quantize(&chunk_float, ggml)?;

                let dst_elem_offset = dst_chunk * elems_per_chunk;
                let dst_data = arenas
                    .get_mut(&dst_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("dst arena {dst_arena} not found")))?
                    .quantized_data_mut()?;
                dst_data.slice_scatter(&chunk_quant, dst_elem_offset)?;
                Ok(())
            }

            // CPU Quant → GPU Float
            (Cpu, KvFormat::Quantized(_), Gpu, KvFormat::Float(dtype)) => {
                let src_data_clone = arenas
                    .get(&src_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("src arena {src_arena} not found")))?
                    .quantized_data()?
                    .clone();

                let dequant = src_data_clone.dequantize(&Device::Cpu)?;
                let total_chunks = dequant.elem_count() / elems_per_chunk;
                let reshaped = dequant.reshape((total_chunks, chunk_size, sub_head_dim))?;
                let chunk_data = reshaped
                    .narrow(0, src_chunk, 1)?
                    .to_device(&dst_device)?
                    .to_dtype(*dtype)?;
                arenas
                    .get_mut(&dst_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("dst arena {dst_arena} not found")))?
                    .float_data_mut()?
                    .slice_set(&chunk_data, 0, dst_chunk)?;
                Ok(())
            }

            // CPU Float → CPU Quant
            (Cpu, KvFormat::Float(_), Cpu, KvFormat::Quantized(_)) => {
                let chunk_float = {
                    let src_data = arenas
                        .get(&src_arena)
                        .ok_or_else(|| {
                            candle::Error::Msg(format!("src arena {src_arena} not found"))
                        })?
                        .float_data()?;
                    src_data.narrow(0, src_chunk, 1)?.squeeze(0)?.contiguous()?
                };

                let ggml = dst_key.format.as_quant().unwrap().to_ggml_dtype();
                let chunk_quant = QTensor::quantize(&chunk_float, ggml)?;

                let dst_elem_offset = dst_chunk * elems_per_chunk;
                let dst_data = arenas
                    .get_mut(&dst_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("dst arena {dst_arena} not found")))?
                    .quantized_data_mut()?;
                dst_data.slice_scatter(&chunk_quant, dst_elem_offset)?;
                Ok(())
            }

            // CPU Quant → CPU Float
            (Cpu, KvFormat::Quantized(_), Cpu, KvFormat::Float(dtype)) => {
                let src_data_clone = arenas
                    .get(&src_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("src arena {src_arena} not found")))?
                    .quantized_data()?
                    .clone();

                let dequant = src_data_clone.dequantize(&Device::Cpu)?;
                let total_chunks = dequant.elem_count() / elems_per_chunk;
                let reshaped = dequant.reshape((total_chunks, chunk_size, sub_head_dim))?;

                let chunk_data = reshaped.narrow(0, src_chunk, 1)?.to_dtype(*dtype)?;
                arenas
                    .get_mut(&dst_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("dst arena {dst_arena} not found")))?
                    .float_data_mut()?
                    .slice_set(&chunk_data, 0, dst_chunk)?;
                Ok(())
            }

            // GPU Quant(R16) → GPU Quant(non-R16): R16 stores raw F16 data in
            // token-oriented [D, T] layout. Dequantize to float, then quantize
            // to the target format via the same path as Float→Quant.
            #[cfg(feature = "cuda")]
            (Gpu, KvFormat::Quantized(QuantFormat::R16), Gpu, KvFormat::Quantized(_)) => {
                let src_data_clone = arenas
                    .get(&src_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("src arena {src_arena} not found")))?
                    .quantized_data()?
                    .clone();
                let src_elem_offset = src_chunk * elems_per_chunk;
                let device = dst_device;

                // Dequant R16 → F16 in token-oriented layout: (1, sub_head_dim, chunk_size)
                let temp_shape = (1, sub_head_dim, chunk_size);
                let mut temp = Tensor::zeros(temp_shape, DType::F16, &device)?;
                src_data_clone.dequantize_into(&mut temp, src_elem_offset, 0, elems_per_chunk)?;

                // Transpose [1, D/P, T] → [1, T, D/P] for quantize_transposed_into
                let chunk_3d = temp.transpose(1, 2)?.contiguous()?;

                let dst_elem_offset = dst_chunk * elems_per_chunk;
                let dst_data = arenas
                    .get_mut(&dst_arena)
                    .ok_or_else(|| candle::Error::Msg(format!("dst arena {dst_arena} not found")))?
                    .quantized_data_mut()?;
                dst_data.quantize_transposed_into(&chunk_3d, dst_elem_offset)?;
                Ok(())
            }

            // Quant → Quant same format (direct byte copy)
            (_, KvFormat::Quantized(_), _, KvFormat::Quantized(_)) => Self::copy_chunk_data_static(
                arenas, src_arena, src_chunk, dst_arena, dst_chunk, _n_kv_head, chunk_size,
                head_dim,
            ),

            _ => candle::bail!(
                "convert_chunk_data: unsupported conversion from {:?} to {:?}",
                src_key,
                dst_key
            ),
        }
    }

    /// Get raw GPU pointer for a tensor at a given element offset.
    #[cfg(feature = "cuda")]
    pub(super) fn tensor_ptr_at_offset(tensor: &Tensor, elem_offset: usize) -> Result<u64> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use half::{bf16, f16};

        let (storage, layout) = tensor.storage_and_layout();
        let cuda_storage = match &*storage {
            candle::Storage::Cuda(c) => c,
            _ => candle::bail!("tensor_ptr_at_offset: expected CUDA storage"),
        };

        // Get the CudaDevice to access the stream
        let cuda_device = cuda_storage.device();
        let stream = cuda_device.cuda_stream();

        // Account for both the layout offset and the requested element offset
        let total_offset = layout.start_offset() + elem_offset;

        // Handle different dtypes - extract pointer with stream for proper synchronization
        let ptr = match tensor.dtype() {
            candle::DType::F32 => {
                let slice = cuda_storage.as_cuda_slice::<f32>()?;
                let slice = slice.slice(total_offset..);
                let (ptr, _guard) = slice.device_ptr(&stream);
                ptr
            }
            candle::DType::F16 => {
                let slice = cuda_storage.as_cuda_slice::<f16>()?;
                let slice = slice.slice(total_offset..);
                let (ptr, _guard) = slice.device_ptr(&stream);
                ptr
            }
            candle::DType::BF16 => {
                let slice = cuda_storage.as_cuda_slice::<bf16>()?;
                let slice = slice.slice(total_offset..);
                let (ptr, _guard) = slice.device_ptr(&stream);
                ptr
            }
            candle::DType::F8E4M3 => {
                let slice = cuda_storage.as_cuda_slice::<float8::F8E4M3>()?;
                let slice = slice.slice(total_offset..);
                let (ptr, _guard) = slice.device_ptr(&stream);
                ptr
            }
            _ => candle::bail!(
                "tensor_ptr_at_offset: unsupported dtype {:?}",
                tensor.dtype()
            ),
        };
        Ok(ptr)
    }

    /// Get raw GPU pointer for a QTensor at a given byte offset.
    #[cfg(feature = "cuda")]
    pub(super) fn qtensor_ptr_at_byte_offset(qtensor: &QTensor, byte_offset: usize) -> Result<u64> {
        // Use the public cuda_data_ptr() method
        let base_ptr = qtensor
            .cuda_data_ptr()
            .ok_or_else(|| candle::Error::Msg("QTensor is not on CUDA device".to_string()))?;
        Ok(base_ptr + byte_offset as u64)
    }

    /// Aggregate per-block format tags into a single per-chunk format.
    ///
    /// Uses majority vote: the format selected by the most blocks wins.
    /// F16 sentinel (99) votes are counted and if they form the majority:
    /// - With `has_float_fallback=true`: returns `None` (chunk stays float)
    /// - With `has_float_fallback=false`: returns the first (highest fidelity) candidate
    ///
    /// Ties are broken towards higher fidelity (lower candidate index).
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn aggregate_chunk_format(
        tags: &[i32],
        candidates: &[crate::kv_cache::QuantFormat],
        has_float_fallback: bool,
    ) -> Result<Option<crate::kv_cache::QuantFormat>> {
        use candle::quantized::cuda::select_qtype_to_ggml;

        if candidates.is_empty() {
            candle::bail!("aggregate_chunk_format: empty candidate list");
        }

        // Count votes per candidate index. Index `candidates.len()` = F16 sentinel.
        let mut votes = vec![0u32; candidates.len() + 1];
        let f16_idx = candidates.len(); // sentinel index

        for &tag in tags {
            if tag == SELECT_FMT_F16 {
                votes[f16_idx] += 1;
                continue;
            }
            let block_ggml = select_qtype_to_ggml(tag)?;
            if let Some(idx) = candidates
                .iter()
                .position(|c| c.to_ggml_dtype() == block_ggml)
            {
                votes[idx] += 1;
            }
            // Unknown tags are ignored
        }

        // Find the candidate with the most votes.
        // On tie, prefer the candidate with the HIGHER index (more aggressive).
        let mut best_idx = 0usize;
        let mut best_count = votes[0];
        for (idx, &count) in votes.iter().enumerate().skip(1) {
            if count > best_count {
                best_count = count;
                best_idx = idx;
            }
        }

        if best_idx == f16_idx {
            if has_float_fallback {
                // Float fallback: chunk stays in its float arena
                Ok(None)
            } else {
                // No float fallback: use highest fidelity quantized candidate
                Ok(Some(candidates[0]))
            }
        } else {
            Ok(Some(candidates[best_idx]))
        }
    }

    /// Write one prepacked chunk directly into the backing's arena storage.
    ///
    /// The provided bytes must already match the backing's configured per-side layout.
    /// This is used for zero-copy restore or benchmark staging where K is already packed
    /// as R16 and V is already packed as F16.
    ///
    /// After writing all chunks for a sequence, the caller must call
    /// `advance_sequence` with the total actual token count.
    /// Restore a sealed chunk from raw byte streams.
    ///
    /// The K/V byte streams contain `n_kv_head × N_PALETTE` palette sub-chunks
    /// concatenated. The pal/scale arguments describe how the encoded bytes
    /// should be interpreted on decode:
    ///
    /// - `k_pal` / `v_pal`: per-head packed palette maps (`n_kv_head × head_dim/4`
    ///   bytes each). Pass an empty `Arc<Vec<u8>>` for identity routing.
    /// - `k_scale` / `v_scale`: per-head outer scales (`n_kv_head × N_PALETTE`
    ///   f32 each). Pass an empty `Arc<Vec<f32>>` for unity (no outer scaling).
    ///
    /// Pass empty Arcs when restoring data that was encoded with neither
    /// palette routing nor outer scaling (e.g. R16 / float captures).
    #[allow(clippy::too_many_arguments)]
    pub fn write_raw_sealed_chunk(
        &self,
        batch_idx: usize,
        block_idx: usize,
        k_bytes: &[u8],
        v_bytes: &[u8],
        k_pal: std::sync::Arc<Vec<u8>>,
        v_pal: std::sync::Arc<Vec<u8>>,
        k_scale: std::sync::Arc<Vec<f32>>,
        v_scale: std::sync::Arc<Vec<f32>>,
    ) -> Result<()> {
        let n_kv_head = self.inner.n_kv_head;
        let chunk_size = CHUNK_SIZE;
        let head_dim = self.inner.head_dim;
        let device = self.inner.device.clone();

        let k_format = self.inner.storage.k_format();
        let v_format = self.inner.storage.v_format();

        // Palette4: each chunk = chunk_size * sub_head_dim (one palette sub-band)
        let sub_head_dim = head_dim / N_PALETTE;
        let elems_per_head = chunk_size * sub_head_dim;

        // k_bytes/v_bytes contain ALL n_kv_head * N_PALETTE palette sub-chunks
        // concatenated: [h0p0, h0p1, h0p2, h0p3, h1p0, ...].
        // Split per-(head, palette) and write each to its own flat chunk.
        let k_head_bytes = match k_format {
            KvFormat::Quantized(qf) => {
                let ggml = qf.to_ggml_dtype();
                let blocks = elems_per_head / ggml.block_size();
                blocks * ggml.type_size()
            }
            KvFormat::Float(dtype) => elems_per_head * dtype.size_in_bytes(),
        };
        let v_head_bytes = match v_format {
            KvFormat::Quantized(qf) => {
                let ggml = qf.to_ggml_dtype();
                let v_head_blocks = elems_per_head / ggml.block_size();
                v_head_blocks * ggml.type_size()
            }
            KvFormat::Float(dtype) => elems_per_head * dtype.size_in_bytes(),
        };
        let expected_k_bytes = n_kv_head * N_PALETTE * k_head_bytes;
        let expected_v_bytes = n_kv_head * N_PALETTE * v_head_bytes;
        if k_bytes.len() != expected_k_bytes {
            candle::bail!(
                "write_raw_sealed_chunk: K byte length mismatch, got {}, expected {}",
                k_bytes.len(),
                expected_k_bytes
            );
        }
        if v_bytes.len() != expected_v_bytes {
            candle::bail!(
                "write_raw_sealed_chunk: V byte length mismatch, got {}, expected {}",
                v_bytes.len(),
                expected_v_bytes
            );
        }

        // Allocate the block's GIDs through the single allocation keystone —
        // it also registers them on the block table with the palettes/scales.
        // This restore path is uniform-format, so every sub-band uses the
        // same K/V format.
        let n_sub = n_kv_head * N_PALETTE;
        let gids = self.alloc_sealed_block(
            batch_idx,
            block_idx,
            &vec![k_format; n_sub],
            &vec![v_format; n_sub],
            k_pal,
            v_pal,
            k_scale,
            v_scale,
        )?;

        // Upload per-(head, palette) byte slices and scatter into flat arena chunks.
        for h in 0..n_kv_head {
            for p in 0..N_PALETTE {
                let slot_idx = h * N_PALETTE + p;
                let k_start = slot_idx * k_head_bytes;
                let k_slice = &k_bytes[k_start..k_start + k_head_bytes];
                let v_start = slot_idx * v_head_bytes;
                let v_slice = &v_bytes[v_start..v_start + v_head_bytes];

                let gid_offset = h * GIDS_PER_HEAD + p * 2;
                let k_gid = &gids.0[gid_offset];
                let v_gid = &gids.0[gid_offset + 1];

                self.inner.storage.try_write(|s| {
                    let k_ai = k_gid.arena_idx();
                    let v_ai = v_gid.arena_idx();
                    match k_format {
                        KvFormat::Quantized(crate::kv_cache::QuantFormat::R16) => {
                            let k_dst = s.arenas_mut()
                                .get_mut(&k_ai)
                                .ok_or_else(|| candle::Error::Msg("k arena not found".into()))?
                                .quantized_data_mut()?;
                            let k_offset = k_gid.chunk_idx() * k_head_bytes;
                            #[cfg(feature = "cuda")]
                            {
                                let dst_ptr = Self::qtensor_ptr_at_byte_offset(k_dst, k_offset)?;
                                unsafe {
                                    candle::cuda_backend::cudarc::driver::result::memcpy_htod_sync(
                                        dst_ptr,
                                        k_slice,
                                    )
                                }
                                .map_err(|e| {
                                    candle::Error::Msg(format!(
                                        "write_raw_sealed_chunk: failed to upload R16 K bytes: {e}"
                                    ))
                                })?;
                            }
                            #[cfg(not(feature = "cuda"))]
                            {
                                // CPU path: raw byte copy into the QStorage backing buffer.
                                // Same unsafe pattern as QStorage::slice_scatter on CPU.
                                match k_dst.storage() {
                                    candle::quantized::QStorage::Cpu(cpu_storage) => {
                                        let dst_ptr = cpu_storage.as_ptr() as *mut u8;
                                        unsafe {
                                            std::ptr::copy_nonoverlapping(k_slice.as_ptr(), dst_ptr.add(k_offset), k_head_bytes);
                                        }
                                    }
                                    _ => candle::bail!("write_raw_sealed_chunk: R16 CPU path requires CPU QStorage"),
                                }
                            }
                        }
                        KvFormat::Quantized(k_qformat) => {
                            let k_qtensor = candle::quantized::ggml_file::qtensor_from_ggml(
                                k_qformat.to_ggml_dtype(),
                                k_slice,
                                vec![elems_per_head],
                                &device,
                            )?;
                            let k_dst = s.arenas_mut()
                                .get_mut(&k_ai)
                                .ok_or_else(|| candle::Error::Msg("k arena not found".into()))?
                                .quantized_data_mut()?;
                            let k_offset = k_gid.chunk_idx() * elems_per_head;
                            k_dst.slice_scatter(&k_qtensor, k_offset)?;
                        }
                        KvFormat::Float(DType::F16) => {
                            let values: Vec<half::f16> = k_slice
                                .chunks_exact(std::mem::size_of::<half::f16>())
                                .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                                .collect();
                            let k_tensor = Tensor::from_vec(
                                values,
                                (1, chunk_size, sub_head_dim),
                                &device,
                            )?;
                            s.arenas_mut()
                                .get_mut(&k_ai)
                                .ok_or_else(|| candle::Error::Msg("k arena not found".into()))?
                                .float_data_mut()?
                                .slice_set(&k_tensor, 0, k_gid.chunk_idx())?;
                        }
                        KvFormat::Float(DType::BF16) => {
                            let values: Vec<half::bf16> = k_slice
                                .chunks_exact(std::mem::size_of::<half::bf16>())
                                .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
                                .collect();
                            let k_tensor = Tensor::from_vec(
                                values,
                                (1, chunk_size, sub_head_dim),
                                &device,
                            )?;
                            s.arenas_mut()
                                .get_mut(&k_ai)
                                .ok_or_else(|| candle::Error::Msg("k arena not found".into()))?
                                .float_data_mut()?
                                .slice_set(&k_tensor, 0, k_gid.chunk_idx())?;
                        }
                        KvFormat::Float(DType::F32) => {
                            let values: Vec<f32> = k_slice
                                .chunks_exact(std::mem::size_of::<f32>())
                                .map(|chunk| {
                                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                                })
                                .collect();
                            let k_tensor = Tensor::from_vec(
                                values,
                                (1, chunk_size, sub_head_dim),
                                &device,
                            )?;
                            s.arenas_mut()
                                .get_mut(&k_ai)
                                .ok_or_else(|| candle::Error::Msg("k arena not found".into()))?
                                .float_data_mut()?
                                .slice_set(&k_tensor, 0, k_gid.chunk_idx())?;
                        }
                        KvFormat::Float(other) => {
                            candle::bail!(
                                "write_raw_sealed_chunk: unsupported float K dtype {other:?}"
                            )
                        }
                    }

                    match v_format {
                        KvFormat::Quantized(crate::kv_cache::QuantFormat::R16) => {
                            let v_dst = s.arenas_mut()
                                .get_mut(&v_ai)
                                .ok_or_else(|| candle::Error::Msg("v arena not found".into()))?
                                .quantized_data_mut()?;
                            let v_offset = v_gid.chunk_idx() * v_head_bytes;
                            #[cfg(feature = "cuda")]
                            {
                                let dst_ptr = Self::qtensor_ptr_at_byte_offset(v_dst, v_offset)?;
                                unsafe {
                                    candle::cuda_backend::cudarc::driver::result::memcpy_htod_sync(
                                        dst_ptr,
                                        v_slice,
                                    )
                                }
                                .map_err(|e| {
                                    candle::Error::Msg(format!(
                                        "write_raw_sealed_chunk: failed to upload R16 V bytes: {e}"
                                    ))
                                })?;
                            }
                            #[cfg(not(feature = "cuda"))]
                            {
                                // CPU path: raw byte copy into the QStorage backing buffer.
                                match v_dst.storage() {
                                    candle::quantized::QStorage::Cpu(cpu_storage) => {
                                        let dst_ptr = cpu_storage.as_ptr() as *mut u8;
                                        unsafe {
                                            std::ptr::copy_nonoverlapping(v_slice.as_ptr(), dst_ptr.add(v_offset), v_head_bytes);
                                        }
                                    }
                                    _ => candle::bail!("write_raw_sealed_chunk: R16 CPU path requires CPU QStorage"),
                                }
                            }
                        }
                        KvFormat::Quantized(v_qformat) => {
                            let v_qtensor = candle::quantized::ggml_file::qtensor_from_ggml(
                                v_qformat.to_ggml_dtype(),
                                v_slice,
                                vec![elems_per_head],
                                &device,
                            )?;
                            let v_dst = s.arenas_mut()
                                .get_mut(&v_ai)
                                .ok_or_else(|| candle::Error::Msg("v arena not found".into()))?
                                .quantized_data_mut()?;
                            let v_offset = v_gid.chunk_idx() * elems_per_head;
                            v_dst.slice_scatter(&v_qtensor, v_offset)?;
                        }
                        KvFormat::Float(DType::F16) => {
                            let values: Vec<half::f16> = v_slice
                                .chunks_exact(std::mem::size_of::<half::f16>())
                                .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                                .collect();
                            let v_tensor = Tensor::from_vec(
                                values,
                                (1, chunk_size, sub_head_dim),
                                &device,
                            )?;
                            s.arenas_mut()
                                .get_mut(&v_ai)
                                .ok_or_else(|| candle::Error::Msg("v arena not found".into()))?
                                .float_data_mut()?
                                .slice_set(&v_tensor, 0, v_gid.chunk_idx())?;
                        }
                        KvFormat::Float(DType::BF16) => {
                            let values: Vec<half::bf16> = v_slice
                                .chunks_exact(std::mem::size_of::<half::bf16>())
                                .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
                                .collect();
                            let v_tensor = Tensor::from_vec(
                                values,
                                (1, chunk_size, sub_head_dim),
                                &device,
                            )?;
                            s.arenas_mut()
                                .get_mut(&v_ai)
                                .ok_or_else(|| candle::Error::Msg("v arena not found".into()))?
                                .float_data_mut()?
                                .slice_set(&v_tensor, 0, v_gid.chunk_idx())?;
                        }
                        KvFormat::Float(DType::F32) => {
                            let values: Vec<f32> = v_slice
                                .chunks_exact(std::mem::size_of::<f32>())
                                .map(|chunk| {
                                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                                })
                                .collect();
                            let v_tensor = Tensor::from_vec(
                                values,
                                (1, chunk_size, sub_head_dim),
                                &device,
                            )?;
                            s.arenas_mut()
                                .get_mut(&v_ai)
                                .ok_or_else(|| candle::Error::Msg("v arena not found".into()))?
                                .float_data_mut()?
                                .slice_set(&v_tensor, 0, v_gid.chunk_idx())?;
                        }
                        KvFormat::Float(other) => {
                            candle::bail!(
                                "write_raw_sealed_chunk: unsupported float V dtype {other:?}"
                            )
                        }
                    }
                    Ok(())
                })?;
            }
        }

        Ok(())
    }

    /// Allocate one sealed block's GIDs — the **allocation keystone** (§16.12).
    ///
    /// This is the allocation half of [`Self::write_raw_sealed_chunk`], with
    /// no byte I/O: it materialises the destination slot, allocates the
    /// per-`(head, palette)` chunk GIDs in arenas of `k_format` / `v_format`,
    /// and registers them on the block table with the chunk's palettes and
    /// scales. It is the **single source of truth** for the GID shape a chunk
    /// of a given `KvFormat` needs.
    ///
    /// The returned [`HeadGids`] addresses freshly-allocated, uninitialised
    /// chunks. Fill them either by uploading raw bytes
    /// (`write_raw_sealed_chunk`) or by snapshotting the slot with
    /// [`Self::record_turn`] and scattering bytes through the
    /// `resolve_sealed_chunk_ptrs` migration path (resume / cold-load).
    #[allow(clippy::too_many_arguments)]
    pub fn alloc_sealed_block(
        &self,
        batch_idx: usize,
        block_idx: usize,
        k_formats: &[KvFormat],
        v_formats: &[KvFormat],
        k_pal: std::sync::Arc<Vec<u8>>,
        v_pal: std::sync::Arc<Vec<u8>>,
        k_scale: std::sync::Arc<Vec<f32>>,
        v_scale: std::sync::Arc<Vec<f32>>,
    ) -> Result<HeadGids> {
        let n_kv_head = self.inner.n_kv_head;
        let want = n_kv_head * N_PALETTE;
        if k_formats.len() != want || v_formats.len() != want {
            candle::bail!(
                "alloc_sealed_block: expected {want} per-sub-band formats, got k={} v={}",
                k_formats.len(),
                v_formats.len()
            );
        }
        let location = self.inner.storage.default_location();

        // Ensure the block table has room for this block index and materialize
        // the destination sequence slot before attaching the fresh GIDs.
        self.ensure_max_blocks(block_idx + 1)?;
        self.ensure_for_offset(batch_idx, block_idx * CHUNK_SIZE, CHUNK_SIZE)?;

        // Allocate GIDS_PER_HEAD * n_kv_head destination chunks, each
        // `(head, palette sub-band)` in its own adaptive `KvFormat`.
        let mut gid_vec: Vec<ChunkGid> = Vec::with_capacity(GIDS_PER_HEAD * n_kv_head);
        {
            let _state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for h in 0..n_kv_head {
                for p in 0..N_PALETTE {
                    let sub = h * N_PALETTE + p;
                    let k_gid =
                        self.alloc_chunk_for_key(ArenaKey::uniform(k_formats[sub], location))?;
                    let v_gid =
                        self.alloc_chunk_for_key(ArenaKey::uniform(v_formats[sub], location))?;
                    gid_vec.push(k_gid);
                    gid_vec.push(v_gid);
                }
            }
        }

        // Register per-head GIDs, palette maps, and outer scales on the block
        // table, then refresh the live decode slot.
        let gids = HeadGids::from_vec(gid_vec);
        let arena_info = self.resolve_arena_info()?;
        self.set_block_gids_sharded_and_update_gpu(
            batch_idx,
            block_idx,
            gids.clone(),
            k_pal,
            v_pal,
            k_scale,
            v_scale,
            &arena_info,
        )?;
        Ok(gids)
    }

    /// Bulk variant of [`Self::alloc_sealed_block`] + [`Self::set_block_window`]
    /// that does all the per-block work for one layer under a **single**
    /// per-layer `state.write()` critical section.
    ///
    /// Per the legacy path, each block paid:
    ///   - 1 `state.write()` for the `alloc_chunk_for_key` loop,
    ///   - 1 `state.write()` for `set_block_gids_sharded_and_update_gpu`,
    ///   - 1 `state.write()` for `set_block_window`,
    ///   - 1 `resolve_arena_info` call (CPU walk over arenas).
    ///
    /// For a Qwen3-30B-A3B turn (~1500 blocks × 48 layers) that was
    /// ~216 000 lock acquisitions and ~72 000 arena-info walks. The bulk
    /// call collapses all of them to 1 acquisition + 1 arena-info walk
    /// per layer (48 each per turn).
    ///
    /// All blocks must satisfy the same per-sub-band shape constraint as
    /// `alloc_sealed_block` (`k_formats.len() == v_formats.len() ==
    /// n_kv_head × N_PALETTE`). Block indices need not be contiguous
    /// but must each be `< max_blocks` after the upfront grow.
    ///
    /// On error, the lock guard drops and any partial allocations remain
    /// in place — the caller is expected to `free_sequence` the slot to
    /// recycle them.
    /// Pre-allocate the slot's block table + sequence chunks for a
    /// cold-load. Caller pairs this with [`Self::alloc_sealed_blocks_bulk`]:
    /// invoke once per layer at the start of the cold-load (with the
    /// turn's total `chunks_per_layer`), then the per-unit bulk allocs
    /// inherit the capacity and skip their own `ensure_*` work
    /// (saving 2 × `state.write()` acquires per bulk call).
    pub fn ensure_capacity_for_blocks(&self, batch_idx: usize, n_blocks: usize) -> Result<()> {
        if n_blocks == 0 {
            return Ok(());
        }
        self.ensure_max_blocks(n_blocks)?;
        self.ensure_for_offset(batch_idx, 0, n_blocks * CHUNK_SIZE)?;
        Ok(())
    }

    /// Returns the alloc'd `HeadGids` alongside the **freshly-resolved**
    /// `arena_info` snapshot used for the GPU metadata push. The caller
    /// can hand both directly to
    /// [`Self::resolve_block_ptrs_from_hgids`] on the same backing —
    /// no `state.read()` lock, no per-block dedup. The arena_info is
    /// valid as long as no arena-mutating call runs in between (true
    /// for the cold-load pipeline's per-unit, per-layer pair).
    pub fn alloc_sealed_blocks_bulk(
        &self,
        batch_idx: usize,
        specs: &[BlockAllocSpec],
        pool_us_out: &mut u64,
        register_us_out: &mut u64,
        gpu_push_us_out: &mut u64,
    ) -> Result<(
        Vec<HeadGids>,
        Vec<crate::kv_cache::arena_table::ResolvedArenaInfo>,
    )> {
        if specs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
        let want = n_kv_head * N_PALETTE;
        for spec in specs {
            if spec.k_formats.len() != want || spec.v_formats.len() != want {
                candle::bail!(
                    "alloc_sealed_blocks_bulk: expected {want} per-sub-band formats, \
                     got k={} v={} (block {})",
                    spec.k_formats.len(),
                    spec.v_formats.len(),
                    spec.block_idx
                );
            }
        }
        let location = self.inner.storage.default_location();

        // **Caller contract**: the slot must already have at least
        // `max(spec.block_idx) + 1` chunks allocated before entering
        // this function. The cold-load pipeline ensures this once up
        // front via [`Self::ensure_capacity_for_blocks`], which avoids
        // the per-call `ensure_max_blocks` + `ensure_for_offset` lock
        // churn (each acquires `state.write()` — 2 acquires/call ×
        // 82 calls/turn = ~10 ms wall-clock on a 1824-chunk turn).

        // `resolve_arena_info` must be called **after** the per-spec
        // gid allocs below (which can register fresh arena indices via
        // `pool.register_arena` when a format's pool is exhausted). We
        // hold off on resolving until just before the GPU metadata
        // push so it reflects every arena that the alloc grew. Doing
        // it earlier would leave the `update_gpu_chunks_bulk` call
        // with stale `base_ptr` / `chunk_byte_stride` for the freshly-
        // registered arenas — silent corruption with `R16`-formatted
        // chunks, or "arena index N out of range" downstream.

        // Single critical section covering: pool allocs, block_gid
        // registration, window stamping, then a **single batched**
        // GPU chunk-metadata push at the end (one guard for the
        // whole layer's blocks, so the drop coalesces adjacent
        // indices into a few `memcpy_htod` runs instead of one per
        // block).
        // ── Phase 1: batched GID alloc + scatter into per-spec slots ─
        //
        // For each (spec, sub-band, K|V) gid, record its **destination**
        // (s_idx, gid_idx) under its `ArenaKey`. Then allocate per-key
        // in one batched call and scatter the freshly-claimed gids
        // straight into per-spec `MaybeUninit<ChunkGid>` buffers.
        //
        // Phase 2 below just hands each filled buffer to
        // `HeadGids::from_vec` — **no inner stitch loop, no HashMap
        // lookups in the locked section**. The cold-load shape (~600
        // gids/spec × ~22 specs/call) used to spend ~1,400 HashMap
        // lookups + IntoIter::next per call inside `state.write()`.
        //
        // Layout per spec: K(h, p) → gid_idx = sub*2, V(h, p) → sub*2+1,
        // where sub = h*N_PALETTE + p (matches the historical inner
        // push order).
        use std::mem::MaybeUninit;
        let t_pool = std::time::Instant::now();
        let gids_per_spec = GIDS_PER_HEAD * n_kv_head;
        let mut positions_per_key: ahash::HashMap<ArenaKey, Vec<(usize, usize)>> =
            ahash::HashMap::with_capacity_and_hasher(8, ahash::RandomState::new());
        for (s_idx, spec) in specs.iter().enumerate() {
            for h in 0..n_kv_head {
                for p in 0..N_PALETTE {
                    let sub = h * N_PALETTE + p;
                    positions_per_key
                        .entry(ArenaKey::uniform(spec.k_formats[sub], location))
                        .or_default()
                        .push((s_idx, sub * 2));
                    positions_per_key
                        .entry(ArenaKey::uniform(spec.v_formats[sub], location))
                        .or_default()
                        .push((s_idx, sub * 2 + 1));
                }
            }
        }
        // Per-spec uninitialised buffers — `set_len` is free (no
        // per-element init), and every slot is written by the
        // scatter below.
        let mut spec_buffers: Vec<Vec<MaybeUninit<super::gid_pool::ChunkGid>>> = (0..specs.len())
            .map(|_| {
                let mut v: Vec<MaybeUninit<super::gid_pool::ChunkGid>> =
                    Vec::with_capacity(gids_per_spec);
                // SAFETY: every slot is written by the per-key scatter
                // below before any read. The Vec is consumed into an
                // initialised `Vec<ChunkGid>` once the scatter is
                // complete.
                unsafe { v.set_len(gids_per_spec) };
                v
            })
            .collect();
        for (key, positions) in positions_per_key {
            let gids = self.alloc_chunks_for_key_bulk(key, positions.len())?;
            for (gid, (s_idx, gid_idx)) in gids.into_iter().zip(positions) {
                spec_buffers[s_idx][gid_idx].write(gid);
            }
        }
        // Phase 1 wrote every position in every spec buffer (the
        // position lists were built to cover exactly one slot per
        // (spec, sub, K|V) tuple). Reinterpret each buffer as a
        // fully-initialised `Vec<ChunkGid>`.
        let per_spec_gids: Vec<Vec<super::gid_pool::ChunkGid>> = spec_buffers
            .into_iter()
            .map(|buf| {
                let mut buf = std::mem::ManuallyDrop::new(buf);
                let ptr = buf.as_mut_ptr() as *mut super::gid_pool::ChunkGid;
                let len = buf.len();
                let cap = buf.capacity();
                // SAFETY: every position in `buf` was written in
                // Phase 1; the layout of `MaybeUninit<T>` matches `T`.
                unsafe { Vec::from_raw_parts(ptr, len, cap) }
            })
            .collect();
        *pool_us_out += t_pool.elapsed().as_micros() as u64;

        // ── Phase 2: register blocks under the state write lock ──────
        let mut hgids_per_block: Vec<HeadGids> = Vec::with_capacity(specs.len());
        let mut updated_blocks: Vec<usize> = Vec::with_capacity(specs.len());
        // Declared outside the `state.write()` scope so we can return
        // it to the caller (immediately reused by
        // `resolve_block_ptrs_from_hgids` to skip its own arena walk).
        let arena_info: Vec<crate::kv_cache::arena_table::ResolvedArenaInfo>;
        // Per-block snapshots (blk + frozen fields) for the resident-record build,
        // captured under the state lock but built/uploaded AFTER it is released so
        // the device upload never serializes the per-forward readers.
        type RecordSrc = (
            usize,
            HeadGids,
            std::sync::Arc<Vec<u8>>,
            std::sync::Arc<Vec<u8>>,
            std::sync::Arc<Vec<f32>>,
            std::sync::Arc<Vec<f32>>,
        );
        let mut record_srcs: Vec<RecordSrc> = Vec::new();
        let mut per_spec_iter = per_spec_gids.into_iter();
        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

            let t_register = std::time::Instant::now();
            for spec in specs {
                let gid_vec = per_spec_iter
                    .next()
                    .expect("per_spec_gids has one entry per spec");
                let gids = HeadGids::from_vec(gid_vec);

                // Register on the block table + stamp the window.
                // GPU chunk-metadata push is deferred until all
                // blocks have been registered (see batched call
                // below).
                if let Some(slot) = state.sequences.get_mut(batch_idx).and_then(|s| s.as_mut()) {
                    slot.set_block_gids(
                        spec.block_idx,
                        gids.clone(),
                        std::sync::Arc::clone(&spec.k_pal),
                        std::sync::Arc::clone(&spec.v_pal),
                        std::sync::Arc::clone(&spec.k_scale),
                        std::sync::Arc::clone(&spec.v_scale),
                    );
                    if let Some(cw) = slot.chunk_at_mut(spec.block_idx) {
                        cw.offset = spec.offset;
                        cw.usage = spec.usage;
                    }
                    updated_blocks.push(spec.block_idx);
                }
                hgids_per_block.push(gids);
            }
            *register_us_out += t_register.elapsed().as_micros() as u64;

            // ── Phase 3: resolve arena_info + GPU metadata push ─────
            //
            // Resolve arena_info NOW — after every spec's allocs have
            // potentially registered new arenas — so the GPU metadata
            // push reflects every arena the chunks point into. (See
            // the comment block before the critical section.)
            let t_gpu_push = std::time::Instant::now();
            arena_info = self.resolve_arena_info()?;
            if let Some(slot) = state.sequences.get_mut(batch_idx).and_then(|s| s.as_mut()) {
                slot.update_gpu_chunks_bulk(&updated_blocks, n_kv_head, head_dim, &arena_info)?;
            }
            *gpu_push_us_out += t_gpu_push.elapsed().as_micros() as u64;

            // Snapshot the frozen fields of each loaded chunk while we hold the
            // lock; the actual record serialize + device upload happens below,
            // off the lock.
            if let Some(slot) = state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                for &blk in &updated_blocks {
                    if let Some(cw) = slot.chunk_at(blk) {
                        record_srcs.push((
                            blk,
                            cw.gids.clone(),
                            cw.k_pal.clone(),
                            cw.v_pal.clone(),
                            cw.k_scale.clone(),
                            cw.v_scale.clone(),
                        ));
                    }
                }
            }
        }

        // Build resident KV-head records at the loaded chunks' GPU placement.
        // Cold-load (and warm→hot elevate that routes through here) is the point
        // at which these sealed chunks become device-resident; building the record
        // once here lets every subsequent reproject glue forward read it via
        // `kvheads_ptr` instead of rebuilding heads per layer. Done OFF the state
        // lock and as a single coalesced upload.
        if !record_srcs.is_empty() {
            let refs: Vec<(&HeadGids, &[u8], &[u8], &[f32], &[f32])> = record_srcs
                .iter()
                .map(|(_, g, kp, vp, ks, vs)| {
                    (
                        g,
                        kp.as_slice(),
                        vp.as_slice(),
                        ks.as_slice(),
                        vs.as_slice(),
                    )
                })
                .collect();
            let metas = self.build_meta_records(&refs, &arena_info)?;
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            if let Some(slot) = state.sequences.get_mut(batch_idx).and_then(|s| s.as_mut()) {
                for ((blk, ..), meta) in record_srcs.iter().zip(metas) {
                    if let Some(cw) = slot.chunk_at_mut(*blk) {
                        cw.meta = meta;
                    }
                }
            }
        }
        Ok((hgids_per_block, arena_info))
    }

    /// Walk a [`HeadGids`] and produce the per-`(head, palette)` K/V
    /// format pair — `(k_formats, v_formats)`, each `n_kv_head × N_PALETTE`
    /// entries in `[h*N_PALETTE + p]` order.
    ///
    /// Adaptive quantization picks a format independently per sub-band, so
    /// persistence must record the whole map (not just head-0) to reallocate
    /// the chunk's arenas faithfully. This is called at seal time to
    /// populate the `k_formats` / `v_formats` snapshot stored on
    /// [`SealedChunk`] — the persist path then reads those directly rather
    /// than re-walking arena state.
    pub fn kv_formats_for_gids(
        &self,
        gids: &super::head_gids::HeadGids,
    ) -> candle::Result<(
        Vec<crate::kv_cache::KvFormat>,
        Vec<crate::kv_cache::KvFormat>,
    )> {
        let n_kv_head = self.inner.n_kv_head;
        self.inner.storage.read(|s| {
            let fmt = |arena_idx: usize, side: &str| {
                s.arena_key(arena_idx).map(|key| key.format).ok_or_else(|| {
                    candle::Error::Msg(format!("{side} arena {arena_idx} not found"))
                })
            };
            let mut k_formats = Vec::with_capacity(n_kv_head * N_PALETTE);
            let mut v_formats = Vec::with_capacity(n_kv_head * N_PALETTE);
            for h in 0..n_kv_head {
                for p in 0..N_PALETTE {
                    k_formats.push(fmt(gids.k_gid_pal(h, p).arena_idx(), "k")?);
                    v_formats.push(fmt(gids.v_gid_pal(h, p).arena_idx(), "v")?);
                }
            }
            Ok((k_formats, v_formats))
        })?
    }

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
                            .map(|k| ArenaKey {
                                format: k.format,
                                location: ArenaLocation::Cpu,
                            })
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

        let arena_chunks = arena_gid_stride();

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
                let cpu_key = ArenaKey {
                    format: key.format,
                    location: ArenaLocation::Cpu,
                };
                src_keys.insert(raw, key);
                new_gids.insert(raw, self.alloc_chunk_for_key(cpu_key)?);
            }
        }

        // Phase 2: one `storage.try_write()` for the batch — every
        // per-chunk data copy runs back-to-back with no relock.
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
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
                let src_key = src_keys[&raw].clone();
                let cpu_key = ArenaKey {
                    format: src_key.format,
                    location: ArenaLocation::Cpu,
                };
                let new_gid = new_gids[&raw].clone();
                if src_key == cpu_key {
                    Self::copy_chunk_data_static(
                        s.arenas_mut(),
                        src_arena_idx,
                        src_chunk_idx,
                        new_gid.arena_idx(),
                        new_gid.chunk_idx(),
                        n_kv_head,
                        CHUNK_SIZE,
                        head_dim,
                    )?;
                } else {
                    Self::convert_chunk_data_static(
                        s.arenas_mut(),
                        src_arena_idx,
                        src_chunk_idx,
                        src_key,
                        new_gid.arena_idx(),
                        new_gid.chunk_idx(),
                        cpu_key,
                        n_kv_head,
                        CHUNK_SIZE,
                        head_dim,
                    )?;
                }
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
        copy_stream: &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>,
        pinned_scratch: &mut Option<candle::quantized::pinned_staging::PinnedBuf>,
        sequences: &[&SealedSequence],
    ) -> candle::Result<Vec<SealedSequence>> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::WrapErr;
        use candle::quantized::pinned_staging::PinnedBuf;

        use super::migrate::{kv_migrate_on, MigrationPlan};

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
        let arena_chunks = arena_gid_stride();
        let arena_info = self.resolve_arena_info()?;
        let mut unique_raws: Vec<i64> = Vec::new();
        let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
        let mut src_ptrs: Vec<(i64, i64)> = Vec::new();
        let mut gid_byte_range: std::collections::HashMap<i64, (usize, usize)> =
            std::collections::HashMap::new();
        let mut src_keys: std::collections::HashMap<i64, ArenaKey> =
            std::collections::HashMap::new();
        let mut total_bytes: usize = 0;

        for seq in sequences {
            for chunk in &seq.chunks {
                for gid in chunk.gids.0.iter() {
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
                    let len = info.chunk_byte_stride as usize;
                    let ptr = info.base_ptr as i64 + chunk_idx as i64 * info.chunk_byte_stride;
                    src_ptrs.push((ptr, info.chunk_byte_stride));
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

        // ── Phase 1: device-side gather on the copy stream ─────────────
        let staging: candle::cuda_backend::cudarc::driver::CudaSlice<u8> =
            unsafe { copy_stream.alloc::<u8>(total_bytes).w()? };
        let staging_base = {
            let (p, _g) = staging.device_ptr(copy_stream);
            p as i64
        };
        let mut plan = MigrationPlan::new();
        let mut off = 0i64;
        for &(ptr, len) in &src_ptrs {
            plan.push(ptr, staging_base + off, len);
            off += len;
        }
        kv_migrate_on(device, &plan, Some(copy_stream))?;
        // ── Phase 2: single DtoH staging → pinned host scratch ─────────
        {
            let scratch = pinned_scratch
                .as_mut()
                .expect("pinned scratch allocated above");
            let dst = &mut scratch.as_mut_slice()[..total_bytes];
            copy_stream.memcpy_dtoh(&staging, dst).w()?;
            copy_stream.synchronize().w()?;
        }
        drop(staging);

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
                let cpu_key = ArenaKey {
                    format: src_key.format,
                    location: ArenaLocation::Cpu,
                };
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

    /// Cross-layer batched hot→warm DtoH. The per-layer
    /// [`Self::migrate_sealed_to_cpu_batch_async`] issues a gather + DtoH + sync
    /// **per backing** (48× per persist pass on Qwen3); this resolves EVERY
    /// layer's chunks into ONE contiguous staging blob, runs a SINGLE gather +
    /// SINGLE DtoH + SINGLE `copy_stream.synchronize()`, then scatters per-layer
    /// into fresh CPU arenas. Collapses the WDDM per-launch/per-sync overhead
    /// that made `copy_ms` the hot→warm drain bottleneck. `backings[i]` owns
    /// `per_layer_seqs[i]`; returns the warm (CPU) sequences per layer, in the
    /// same order. Bit-identical to the per-layer path — proven by
    /// `migrate_layers_deferred_matches_immediate_bytes` extended, and the
    /// dedicated `migrate_layers_batched_matches_per_layer_bytes`.
    #[cfg(feature = "cuda")]
    pub fn migrate_sealed_layers_to_cpu_batch(
        backings: &[&ChunkedKvBacking],
        device: &Device,
        copy_stream: &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>,
        pinned_scratch: &mut Option<candle::quantized::pinned_staging::PinnedBuf>,
        per_layer_seqs: &[Vec<&SealedSequence>],
    ) -> candle::Result<Vec<Vec<SealedSequence>>> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::WrapErr;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::collections::{HashMap, HashSet};

        use super::migrate::{kv_migrate_on, MigrationPlan};

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
        // per staging batch below (not globally), so the transient GPU staging
        // buffer never scales with the whole cross-layer batch.
        struct LayerResolve {
            unique_raws: Vec<i64>,
            /// Source device ptr + byte length per unique raw gid.
            src: HashMap<i64, (i64, usize)>,
            src_keys: HashMap<i64, ArenaKey>,
            layer_bytes: usize,
        }

        // ── Resolve every layer's sources (device ptr + len per gid) ────
        let arena_chunks = arena_gid_stride();
        let mut layers: Vec<LayerResolve> = Vec::with_capacity(backings.len());
        let mut grand_total: usize = 0;
        for (li, backing) in backings.iter().enumerate() {
            let arena_info = backing.resolve_arena_info()?;
            let mut seen: HashSet<i64> = HashSet::new();
            let mut unique_raws: Vec<i64> = Vec::new();
            let mut src: HashMap<i64, (i64, usize)> = HashMap::new();
            let mut src_keys: HashMap<i64, ArenaKey> = HashMap::new();
            let mut layer_bytes = 0usize;
            for seq in &per_layer_seqs[li] {
                for chunk in &seq.chunks {
                    for gid in chunk.gids.0.iter() {
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
                        let len = info.chunk_byte_stride as usize;
                        let ptr = info.base_ptr as i64 + chunk_idx as i64 * info.chunk_byte_stride;
                        src.insert(raw, (ptr, len));
                        unique_raws.push(raw);
                        layer_bytes += len;
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
            grand_total += layer_bytes;
            layers.push(LayerResolve {
                unique_raws,
                src,
                src_keys,
                layer_bytes,
            });
        }

        if grand_total == 0 {
            return Ok(per_layer_seqs
                .iter()
                .map(|seqs| seqs.iter().map(|s| (*s).clone()).collect())
                .collect());
        }

        // ── Migrate in contiguous layer batches, each capped by STAGING_CAP ─
        // A single cross-layer gather would size the transient GPU staging to
        // the ENTIRE batch (all layers at once) — under ingest pressure that
        // alloc can OOM the very copy meant to relieve pressure. Instead, group
        // layers so each batch's staging stays ≤ cap and issue ONE gather + DtoH
        // + sync per batch. Layers are ~uniform (≈1/n_layers of the total each),
        // so this is ceil(total/cap) syncs — still collapsing the WDDM per-launch
        // overhead that made `copy_ms` the drain bottleneck, not the per-layer 48.
        const STAGING_CAP_BYTES: usize = MIGRATION_STAGING_CAP_BYTES;
        let mut out: Vec<Vec<SealedSequence>> = Vec::with_capacity(backings.len());
        let mut li = 0usize;
        while li < backings.len() {
            // Grow the batch while under the cap; always take ≥1 layer so a lone
            // layer larger than the cap still makes progress.
            let mut batch_bytes = 0usize;
            let mut lj = li;
            while lj < backings.len() {
                let lb = layers[lj].layer_bytes;
                if lj > li && batch_bytes + lb > STAGING_CAP_BYTES {
                    break;
                }
                batch_bytes += lb;
                lj += 1;
            }

            if batch_bytes == 0 {
                // A trailing run of wholly-empty layers — emit their (empty)
                // sequences with no GPU work.
                for seqs in &per_layer_seqs[li..lj] {
                    out.push(seqs.iter().map(|s| (*s).clone()).collect());
                }
                li = lj;
                continue;
            }

            // Allocate the host + GPU staging for [li, lj), shrinking the batch
            // and retrying on OOM. Both allocs happen BEFORE any scatter side
            // effect, so a failed attempt leaves no partial state — halving the
            // layer span (down to a single ~30 MB layer) lets migration make
            // progress even when host RAM is fragmented or VRAM is tight, instead
            // of aborting. Only when a lone layer still won't fit do we propagate
            // the error (the turn stays hot-float + consistent, retried next pass).
            let (staging, batch_bytes) = loop {
                let batch_bytes: usize = layers[li..lj].iter().map(|l| l.layer_bytes).sum();
                // Host scratch (fallible — see the per-layer variant's note).
                let host_res = {
                    let need_grow = pinned_scratch
                        .as_ref()
                        .map(|b| b.len() < batch_bytes)
                        .unwrap_or(true);
                    if need_grow {
                        PinnedBuf::alloc_default_or_host_fallible(batch_bytes)
                            .map(|b| *pinned_scratch = Some(b))
                    } else {
                        Ok(())
                    }
                };
                // GPU staging (only once the host scratch is in place).
                let alloc_res =
                    host_res.and_then(|_| unsafe { copy_stream.alloc::<u8>(batch_bytes) }.w());
                match alloc_res {
                    Ok(s) => break (s, batch_bytes),
                    Err(e) if lj > li + 1 => {
                        lj = li + ((lj - li) / 2).max(1);
                        tracing::warn!(
                            target: "candle_conversation::persistence::tier",
                            batch_bytes,
                            layers = lj - li,
                            "hot→warm staging alloc failed under memory pressure; \
                             shrinking migration batch and retrying: {e}"
                        );
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            };

            // ── ONE gather + ONE DtoH + ONE sync for this batch ─────────────
            let staging_base = {
                let (p, _g) = staging.device_ptr(copy_stream);
                p as i64
            };
            let mut plan = MigrationPlan::new();
            // Batch-local byte offset per unique gid, per layer, for the scatter.
            let mut batch_ranges: Vec<HashMap<i64, (usize, usize)>> = Vec::with_capacity(lj - li);
            let mut off = 0i64;
            for layer in &layers[li..lj] {
                let mut ranges: HashMap<i64, (usize, usize)> = HashMap::new();
                for &raw in &layer.unique_raws {
                    let (ptr, len) = layer.src[&raw];
                    plan.push(ptr, staging_base + off, len as i64);
                    ranges.insert(raw, (off as usize, len));
                    off += len as i64;
                }
                batch_ranges.push(ranges);
            }
            kv_migrate_on(device, &plan, Some(copy_stream))?;
            {
                let scratch = pinned_scratch
                    .as_mut()
                    .expect("pinned scratch allocated above");
                let dst = &mut scratch.as_mut_slice()[..batch_bytes];
                copy_stream.memcpy_dtoh(&staging, dst).w()?;
                copy_stream.synchronize().w()?;
            }
            drop(staging);

            // ── Per-layer scatter into fresh CPU arenas ─────────────────────
            for (k, backing) in backings[li..lj].iter().enumerate() {
                let resolve = &layers[li + k];
                let ranges = &batch_ranges[k];
                // Phase 3: allocate dest CPU GIDs (one state.write per backing).
                let mut new_gids: HashMap<i64, ChunkGid> = HashMap::new();
                {
                    let _state = backing
                        .state
                        .write()
                        .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
                    for &raw in &resolve.unique_raws {
                        let src_key = &resolve.src_keys[&raw];
                        let cpu_key = ArenaKey {
                            format: src_key.format,
                            location: ArenaLocation::Cpu,
                        };
                        new_gids.insert(raw, backing.alloc_chunk_for_key(cpu_key)?);
                    }
                }
                // Phase 4: write this layer's slice of the batch blob → CPU arena.
                let scratch_ref = pinned_scratch
                    .as_ref()
                    .expect("pinned scratch allocated above");
                backing.inner.storage.try_write(|s| -> candle::Result<()> {
                    for &raw in &resolve.unique_raws {
                        let (off, len) = ranges[&raw];
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
                // Phase 5: rebuild SealedSequences with mapped CPU GIDs.
                let seqs: candle::Result<Vec<SealedSequence>> = per_layer_seqs[li + k]
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
            li = lj;
        }
        Ok(out)
    }

    /// Write one chunk-slot's worth of raw bytes into a CPU `Arena::Float`
    /// at `(arena_idx, chunk_idx)`. The bytes are interpreted as the
    /// arena's element dtype; a temp CPU [`Tensor`] is constructed over
    /// them and `slice_set` writes it into the dest chunk slot.
    ///
    /// `Arena::Quantized` is not supported here — see the module-level
    /// caveat on [`Self::migrate_sealed_to_cpu_batch_async`].
    ///
    /// Gated behind the cuda feature to match the only caller
    /// (`migrate_sealed_to_cpu_batch_async`).
    #[cfg(feature = "cuda")]
    fn write_chunk_from_pinned_bytes(
        arenas: &mut ahash::AHashMap<usize, Arena>,
        arena_idx: usize,
        chunk_idx: usize,
        bytes: &[u8],
    ) -> candle::Result<()> {
        let arena = arenas.get_mut(&arena_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "write_chunk_from_pinned_bytes: arena {arena_idx} missing"
            ))
        })?;
        match arena {
            Arena::Float { data, dtype, .. } => {
                let dims = data.dims();
                if dims.is_empty() {
                    return Err(candle::Error::Msg(
                        "write_chunk_from_pinned_bytes: arena tensor has zero dims".into(),
                    ));
                }
                // Per-chunk shape = (1, ...remaining dims of the arena tensor)
                let per_chunk: Vec<usize> = std::iter::once(1usize)
                    .chain(dims[1..].iter().copied())
                    .collect();
                let n_elems: usize = dims[1..].iter().product();
                let expected_bytes = n_elems * dtype.size_in_bytes();
                if bytes.len() != expected_bytes {
                    return Err(candle::Error::Msg(format!(
                        "write_chunk_from_pinned_bytes: arena {arena_idx} expects {expected_bytes} \
                         bytes per chunk, got {}",
                        bytes.len()
                    )));
                }
                let temp = match dtype {
                    DType::BF16 => {
                        // SAFETY: bytes is at least 2-byte aligned (pinned host pages
                        // are page-aligned), length validated above.
                        let slice = unsafe {
                            std::slice::from_raw_parts(bytes.as_ptr() as *const half::bf16, n_elems)
                        };
                        Tensor::from_slice(slice, per_chunk.as_slice(), &Device::Cpu)?
                    }
                    DType::F16 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts(bytes.as_ptr() as *const half::f16, n_elems)
                        };
                        Tensor::from_slice(slice, per_chunk.as_slice(), &Device::Cpu)?
                    }
                    DType::F32 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n_elems)
                        };
                        Tensor::from_slice(slice, per_chunk.as_slice(), &Device::Cpu)?
                    }
                    other => {
                        return Err(candle::Error::Msg(format!(
                            "write_chunk_from_pinned_bytes: unsupported Float dtype {other:?}"
                        )));
                    }
                };
                data.slice_set(&temp, 0, chunk_idx)?;
                Ok(())
            }
            Arena::Quantized { data, .. } => {
                // Quantized arenas are a flat byte slab — `slice_scatter`
                // / `slice_range_copy` already do `ptr::copy_nonoverlapping`
                // under the hood, parameterised on byte offsets. The
                // gather pulled exactly one `chunk_byte_stride` of bytes
                // per chunk slot into pinned scratch, so the destination
                // offset is simply `chunk_idx * bytes.len()`.
                let byte_offset = chunk_idx * bytes.len();
                data.write_bytes_at(byte_offset, bytes)?;
                Ok(())
            }
        }
    }

    /// **Fully-batched** RAM→VRAM migration — the symmetric inverse of
    /// [`Self::migrate_sealed_to_cpu_batch_async`]. Uses the same
    /// dedicated copy stream, the same pinned host scratch, and the
    /// same [`super::migrate::kv_migrate_on`] scatter kernel.
    ///
    /// ```text
    ///   CPU arena chunks ──read_chunk_into_pinned────▶ pinned host scratch
    ///                                                      ┃
    ///                                                      ▼
    ///                                          memcpy_htod (copy stream)
    ///                                                      ┃
    ///                                                      ▼
    ///                                          device staging buffer
    ///                                                      ┃
    ///                                          kv_migrate_on scatter
    ///                                                      ┃
    ///                                                      ▼
    ///                                          fresh GPU arena chunks
    /// ```
    ///
    /// Same locking discipline: one `state.write()` for dest GID
    /// allocation, one `storage.read()` for source byte extraction
    /// (CPU arenas → pinned scratch). The CPU-side extraction is a
    /// dtype-dispatched `Tensor::to_vec` + memcpy — symmetric to the
    /// downward path's `Tensor::from_slice` + `slice_set`.
    ///
    /// Float arenas (BF16/F16/F32) only on this fast path.
    #[cfg(feature = "cuda")]
    pub fn migrate_sealed_to_gpu_batch_async(
        &self,
        device: &Device,
        copy_stream: &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>,
        pinned_scratch: &mut Option<candle::quantized::pinned_staging::PinnedBuf>,
        sequences: &[&SealedSequence],
    ) -> candle::Result<Vec<SealedSequence>> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::WrapErr;
        use candle::quantized::pinned_staging::PinnedBuf;

        use super::migrate::{kv_migrate_on, MigrationPlan};

        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        let cuda_dev = match device {
            candle::Device::Cuda(d) => d,
            _ => {
                return Err(candle::Error::Msg(
                    "migrate_sealed_to_gpu_batch_async requires a CUDA device".into(),
                ))
            }
        };
        let _ = cuda_dev;

        // ── Resolve sources + dedup ─────────────────────────────────────
        // Source arenas live on CPU; we record each unique source GID's
        // byte size by reading the arena's per-chunk shape × element size.
        let arena_chunks = arena_gid_stride();
        let mut unique_raws: Vec<i64> = Vec::new();
        let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
        let mut src_keys: std::collections::HashMap<i64, ArenaKey> =
            std::collections::HashMap::new();
        let mut gid_byte_range: std::collections::HashMap<i64, (usize, usize)> =
            std::collections::HashMap::new();
        let mut total_bytes: usize = 0;

        self.inner.storage.read(|s| -> candle::Result<()> {
            for seq in sequences {
                for chunk in &seq.chunks {
                    for gid in chunk.gids.0.iter() {
                        let raw = gid.raw();
                        if !seen.insert(raw) {
                            continue;
                        }
                        let arena_idx = (raw as usize) / arena_chunks;
                        let key = s.arena_key(arena_idx).ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "migrate_sealed_to_gpu_batch_async: arena {arena_idx} not found"
                            ))
                        })?;
                        if key.location != ArenaLocation::Cpu {
                            return Err(candle::Error::Msg(
                                "migrate_sealed_to_gpu_batch_async: source arena is not \
                                 CPU-resident"
                                    .into(),
                            ));
                        }
                        let arena = s.arenas().get(&arena_idx).ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "migrate_sealed_to_gpu_batch_async: arena {arena_idx} missing"
                            ))
                        })?;
                        let len = chunk_byte_size_of(arena)?;
                        src_keys.insert(raw, key);
                        gid_byte_range.insert(raw, (total_bytes, len));
                        total_bytes += len;
                        unique_raws.push(raw);
                    }
                }
            }
            Ok(())
        })??;

        if total_bytes == 0 {
            return sequences.iter().map(|seq| Ok((*seq).clone())).collect();
        }

        // ── Ensure pinned scratch capacity ──────────────────────────────
        let need_grow = pinned_scratch
            .as_ref()
            .map(|b| b.len() < total_bytes)
            .unwrap_or(true);
        if need_grow {
            *pinned_scratch = Some(match PinnedBuf::alloc_owned(total_bytes) {
                Ok(b) => b,
                Err(_) => PinnedBuf::Host {
                    data: vec![0u8; total_bytes],
                },
            });
        }

        // ── Phase 1: read CPU arena bytes into pinned scratch ──────────
        // One storage.read() across all chunks. Per-chunk we do
        // dtype-dispatched Tensor::to_vec + memcpy — symmetric to the
        // downward path's per-chunk Tensor::from_slice + slice_set.
        let scratch = pinned_scratch
            .as_mut()
            .expect("pinned scratch allocated above");
        self.inner.storage.read(|s| -> candle::Result<()> {
            let dst_all = scratch.as_mut_slice();
            for &raw in &unique_raws {
                let arena_idx = (raw as usize) / arena_chunks;
                let chunk_idx = (raw as usize) % arena_chunks;
                let (off, len) = gid_byte_range[&raw];
                let arena = s.arenas().get(&arena_idx).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "migrate_sealed_to_gpu_batch_async: arena {arena_idx} missing"
                    ))
                })?;
                read_chunk_into_pinned_bytes(arena, chunk_idx, &mut dst_all[off..off + len])?;
            }
            Ok(())
        })??;

        // ── Phase 2: HtoD pinned → device staging on the copy stream ───
        let src_bytes = &scratch.as_slice()[..total_bytes];
        let mut staging: candle::cuda_backend::cudarc::driver::CudaSlice<u8> =
            unsafe { copy_stream.alloc::<u8>(total_bytes).w()? };
        copy_stream.memcpy_htod(src_bytes, &mut staging).w()?;
        let staging_base = {
            let (p, _g) = staging.device_ptr(copy_stream);
            p as i64
        };

        // ── Phase 3: allocate dest GPU GIDs (one state.write) ──────────
        let mut new_gids: std::collections::HashMap<i64, ChunkGid> =
            std::collections::HashMap::new();
        {
            let _state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for &raw in &unique_raws {
                let src_key = &src_keys[&raw];
                let gpu_key = ArenaKey {
                    format: src_key.format,
                    location: ArenaLocation::Gpu,
                };
                new_gids.insert(raw, self.alloc_chunk_for_key(gpu_key)?);
            }
        }

        // Stream-ordered allocation fence: the dest arenas above come from the
        // device's ASYNC pool, whose allocations become valid in primary-stream
        // order — but every access below runs on the COPY stream. With no
        // recorded dependency, the copy stream can reach the new arena bytes
        // before the primary stream reaches the allocation point (routine under
        // bulk load, where waves queue the primary stream seconds deep) —
        // CUDA_ERROR_ILLEGAL_ADDRESS on the scatter/convert. Drain the device
        // once so the allocations are valid on every stream before any
        // copy-stream use.
        device.synchronize()?;

        // ── Phase 4: build scatter plan staging → dest GPU arenas ──────
        let gpu_arena_info = self.resolve_arena_info()?;
        let mut plan = MigrationPlan::new();
        for &raw in &unique_raws {
            let new_gid = new_gids[&raw].clone();
            let dst_arena_idx = new_gid.arena_idx();
            let dst_chunk_idx = new_gid.chunk_idx();
            let info = gpu_arena_info.get(dst_arena_idx).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "migrate_sealed_to_gpu_batch_async: dest arena {dst_arena_idx} out of range"
                ))
            })?;
            if info.base_ptr == 0 {
                return Err(candle::Error::Msg(
                    "migrate_sealed_to_gpu_batch_async: dest arena not GPU-resident".into(),
                ));
            }
            let dst_ptr = info.base_ptr as i64 + dst_chunk_idx as i64 * info.chunk_byte_stride;
            let (off, len) = gid_byte_range[&raw];
            plan.push(staging_base + off as i64, dst_ptr, len as i64);
        }

        // ── Phase 5: device-side scatter on the copy stream ────────────
        kv_migrate_on(device, &plan, Some(copy_stream))?;
        drop(staging);

        // ── Phase 6: rebuild SealedSequences with mapped GIDs ──────────
        // Warm→hot elevate rebuilds each chunk's resident KV-head record at its
        // new GPU placement, batched per sequence (one coalesced upload), so a
        // subsequent reproject glue forward reads it instead of rebuilding heads.
        sequences
            .iter()
            .map(|seq| -> candle::Result<SealedSequence> {
                let mapped: Vec<HeadGids> = seq
                    .chunks
                    .iter()
                    .map(|chunk| {
                        chunk
                            .gids
                            .map_unique(|gid| Ok(new_gids[&gid.raw()].clone()))
                    })
                    .collect::<candle::Result<_>>()?;
                let refs: Vec<(&HeadGids, &[u8], &[u8], &[f32], &[f32])> = seq
                    .chunks
                    .iter()
                    .zip(&mapped)
                    .map(|(chunk, m)| {
                        (
                            m,
                            chunk.k_pal.as_slice(),
                            chunk.v_pal.as_slice(),
                            chunk.k_scale.as_slice(),
                            chunk.v_scale.as_slice(),
                        )
                    })
                    .collect();
                let metas = self.build_meta_records(&refs, &gpu_arena_info)?;
                let new_chunks: Vec<SealedChunk> = seq
                    .chunks
                    .iter()
                    .zip(mapped)
                    .zip(metas)
                    .map(|((chunk, m), meta)| SealedChunk {
                        gids: m,
                        meta,
                        ..chunk.clone()
                    })
                    .collect();
                Ok(SealedSequence {
                    chunks: new_chunks,
                    token_count: seq.token_count,
                    chunk_size: seq.chunk_size,
                    location: ArenaLocation::Gpu,
                })
            })
            .collect()
    }

    /// Migrate one sealed sequence from the CPU (warm) tier back to the GPU (hot) tier.
    ///
    /// Symmetric inverse of [`migrate_sealed_to_cpu`].  Sharing structure is
    /// preserved by [`HeadGids::map_unique`].
    pub fn migrate_sealed_to_gpu(&self, sealed: &SealedSequence) -> candle::Result<SealedSequence> {
        let gpu_chunks: candle::Result<Vec<SealedChunk>> = sealed
            .chunks
            .iter()
            .map(|chunk| {
                let new_gids = chunk.gids.map_unique(|gid| {
                    let gpu_key = self.inner.storage.read(|s| {
                        s.arena_key(gid.arena_idx())
                            .map(|k| ArenaKey {
                                format: k.format,
                                location: ArenaLocation::Gpu,
                            })
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
                    // batched elevate (`migrate_sealed_to_gpu_batch_async`) is the
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
            location: ArenaLocation::Gpu,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::chunked::ChunkedKvBacking;
    use candle::{DType, Device, Tensor};

    /// Build a CPU-backed `ChunkedKvBacking` with a single float BF16 arena.
    fn cpu_backing(n_kv_head: usize, head_dim: usize) -> ChunkedKvBacking {
        ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::BF16, &Device::Cpu, 256).unwrap()
    }

    /// Seed `n_tokens` BF16 ones into slot 0, return a sealed snapshot.
    fn seed_and_seal(
        backing: &ChunkedKvBacking,
        n_kv_head: usize,
        head_dim: usize,
        n_tokens: usize,
    ) -> SealedSequence {
        let slot = backing.alloc_sequence().unwrap();
        backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
        let k = Tensor::ones(
            (1, n_kv_head, n_tokens, head_dim),
            DType::BF16,
            &Device::Cpu,
        )
        .unwrap();
        let v = k.clone();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, n_tokens);
        backing.record_turn(slot).unwrap()
    }

    /// `migrate_sealed_to_cpu` on a CPU-backed sequence copies chunks into new
    /// CPU arena slots.  Structural invariants (token count, chunk count, location)
    /// must be preserved.
    #[test]
    fn migrate_to_cpu_preserves_structure() {
        let n_kv_head = 4;
        let head_dim = 32;
        let n_tokens = 40; // 1 full chunk (32) + 1 partial (8)
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, n_tokens);

        assert_eq!(sealed.token_count, n_tokens);
        assert_eq!(sealed.chunks.len(), 2);

        let cpu = backing.migrate_sealed_to_cpu(&sealed).unwrap();

        assert_eq!(cpu.location, ArenaLocation::Cpu);
        assert_eq!(cpu.token_count, n_tokens);
        assert_eq!(cpu.chunk_size, sealed.chunk_size);
        assert_eq!(cpu.chunks.len(), sealed.chunks.len());

        // Token counts per chunk must be preserved.
        for (orig, mig) in sealed.chunks.iter().zip(cpu.chunks.iter()) {
            assert_eq!(mig.token_count, orig.token_count);
            assert_eq!(mig.offset, orig.offset);
        }
    }

    /// After migration, the migrated GIDs must be distinct from the originals
    /// (new physical chunks were allocated, not just re-wrapped).
    #[test]
    fn migrate_to_cpu_produces_distinct_gids() {
        let n_kv_head = 2;
        let head_dim = 16;
        let n_tokens = 32; // exactly one full chunk
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, n_tokens);

        let cpu = backing.migrate_sealed_to_cpu(&sealed).unwrap();

        assert_eq!(cpu.chunks.len(), 1);
        // At least one GID slot must differ from the original.
        let orig_gids: Vec<i64> = sealed.chunks[0].gids.iter().map(|g| g.raw()).collect();
        let mig_gids: Vec<i64> = cpu.chunks[0].gids.iter().map(|g| g.raw()).collect();
        assert_ne!(
            orig_gids, mig_gids,
            "migrated chunks must be new physical allocations"
        );
    }

    /// Migration preserves the GID-sharing structure of the source HeadGids.
    /// Slots that share a raw id in the source share a raw id in the output
    /// (same degree of uniqueness), which means shared physical chunks are
    /// migrated once, not once-per-slot.
    #[test]
    fn migrate_preserves_gid_sharing_structure() {
        let n_kv_head = 1;
        let head_dim = 16;
        let n_tokens = 32;
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, n_tokens);

        let unique_src: std::collections::HashSet<i64> =
            sealed.chunks[0].gids.iter().map(|g| g.raw()).collect();

        let cpu = backing.migrate_sealed_to_cpu(&sealed).unwrap();

        let unique_dst: std::collections::HashSet<i64> =
            cpu.chunks[0].gids.iter().map(|g| g.raw()).collect();

        assert_eq!(
            unique_src.len(),
            unique_dst.len(),
            "sharing structure must be preserved across migration"
        );
    }

    /// Round-trip: CPU → GPU migration (simulated as CPU→CPU on a CPU-only backing)
    /// preserves structure the same way as the forward direction.
    #[test]
    fn migrate_to_gpu_preserves_structure() {
        let n_kv_head = 2;
        let head_dim = 32;
        let n_tokens = 64; // exactly two full chunks
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, n_tokens);

        // On a CPU backing, "to_gpu" migrates to the same CPU location
        // (no CUDA device present).  This tests the code path without CUDA.
        let cpu_to_gpu = backing.migrate_sealed_to_gpu(&sealed).unwrap();

        assert_eq!(cpu_to_gpu.token_count, n_tokens);
        assert_eq!(cpu_to_gpu.chunks.len(), 2);
        for (orig, mig) in sealed.chunks.iter().zip(cpu_to_gpu.chunks.iter()) {
            assert_eq!(mig.token_count, orig.token_count);
        }
    }

    /// Empty sequence migrates without error.
    #[test]
    fn migrate_empty_sequence() {
        let backing = cpu_backing(2, 16);
        let empty = SealedSequence {
            chunks: vec![],
            token_count: 0,
            chunk_size: CHUNK_SIZE,
            location: ArenaLocation::Cpu,
        };
        let cpu = backing.migrate_sealed_to_cpu(&empty).unwrap();
        assert_eq!(cpu.chunks.len(), 0);
        assert_eq!(cpu.token_count, 0);

        let gpu = backing.migrate_sealed_to_gpu(&empty).unwrap();
        assert_eq!(gpu.chunks.len(), 0);
    }

    // ── Quantized offset correctness ─────────────────────────────────────────
    //
    // These tests exercise the `elems_per_chunk = chunk_size * sub_head_dim`
    // calculation in `copy_chunk_data_static` and `convert_chunk_data_static`.
    //
    // Arenas are allocated with shape (arena_chunks, CHUNK_SIZE, sub_head_dim)
    // where sub_head_dim = head_dim / N_PALETTE.  Using `head_dim` instead of
    // `sub_head_dim` produces a 4× offset overrun on N_PALETTE=4 models.
    //
    // The CPU tests exercise this without requiring CUDA by migrating Float→Q8_0
    // and Q8_0→Float.  When `elems_per_chunk` is wrong the reverse reshape gives
    // inner dimension `head_dim` (e.g. 32) instead of `sub_head_dim` (e.g. 8),
    // causing `slice_set` to fail with a shape mismatch.

    fn cpu_quant_key() -> ArenaKey {
        ArenaKey::uniform(
            crate::kv_cache::KvFormat::Quantized(crate::kv_cache::QuantFormat::Q8_0),
            ArenaLocation::Cpu,
        )
    }

    fn cpu_float_key() -> ArenaKey {
        ArenaKey::uniform(
            crate::kv_cache::KvFormat::Float(candle::DType::BF16),
            ArenaLocation::Cpu,
        )
    }

    /// Float → Q8_0 → Float round-trip must preserve chunk shapes.
    ///
    /// head_dim=128 → sub_head_dim=32 (= CHUNK_SIZE, exact Q8_0 block fit).
    /// With wrong `elems_per_chunk = chunk_size * head_dim`, the Q8_0 → Float
    /// dequantize + reshape produces inner dimension `head_dim` (128) instead of
    /// `sub_head_dim` (32), causing `slice_set` to fail with a shape mismatch.
    /// This directly detects the N_PALETTE offset bug without needing CUDA.
    #[test]
    fn float_to_quant_roundtrip_sub_head_dim() {
        let n_kv_head = 2;
        let head_dim = 128; // sub_head_dim = head_dim / N_PALETTE = 32 (= CHUNK_SIZE)
        let n_tokens = 64; // two full chunks
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, n_tokens);
        assert_eq!(sealed.chunks.len(), 2);

        let qkey = cpu_quant_key();
        let fkey = cpu_float_key();

        for chunk in &sealed.chunks {
            let k_gid = chunk.gids.k_gid(0);
            let qgid = backing
                .migrate_chunk(k_gid.raw(), qkey.clone())
                .expect("Float → Q8_0 must succeed");
            backing
                .migrate_chunk(qgid.raw(), fkey.clone())
                .expect("Q8_0 → Float round-trip failed (shape mismatch = wrong sub_head_dim)");
        }
    }

    /// Quant → Quant copy with a non-zero source chunk index exercises
    /// `copy_chunk_data_static` with `src_chunk > 0`.
    ///
    /// Populates 3 Q8_0 slots from three sequential chunks, then copies
    /// the third slot (chunk_idx ≥ 2) to a fresh slot.  A wrong
    /// `elems_per_chunk = chunk_size * head_dim` puts data at the wrong byte
    /// offset; the subsequent Q8_0 → Float reshape then fails with a shape mismatch
    /// (inner dim head_dim=128 vs expected sub_head_dim=32).
    #[test]
    fn quant_to_quant_copy_nonzero_chunk_index() {
        let n_kv_head = 2;
        let head_dim = 128; // sub_head_dim = 32 (= CHUNK_SIZE, exact Q8_0 block fit)
        let n_tokens = 96; // three full chunks
        let backing = cpu_backing(n_kv_head, head_dim);
        let sealed = seed_and_seal(&backing, n_kv_head, head_dim, n_tokens);
        assert_eq!(sealed.chunks.len(), 3);

        let qkey = cpu_quant_key();
        let fkey = cpu_float_key();

        // Populate Q8_0 arena slots 0, 1, 2 from three sequential float chunks.
        let mut quant_gids: Vec<_> = sealed
            .chunks
            .iter()
            .map(|chunk| {
                backing
                    .migrate_chunk(chunk.gids.k_gid(0).raw(), qkey.clone())
                    .expect("Float → Q8_0 for chunk seeding")
            })
            .collect();

        // Copy the 3rd Q8_0 slot (chunk_idx ≥ 2) to a new slot.
        let src_gid = quant_gids.pop().unwrap();
        let copy_gid = backing
            .migrate_chunk(src_gid.raw(), qkey.clone())
            .expect("Q8_0 → Q8_0 copy with non-zero chunk_idx must succeed");

        // Verify the copy can be dequantized back without shape errors.
        backing
            .migrate_chunk(copy_gid.raw(), fkey.clone())
            .expect("Q8_0 → Float after copy must succeed (shape error = wrong sub_head_dim)");
    }

    // ── Batched-async migrate tests (GPU-only) ──────────────────────────────
    //
    // These exercise `migrate_sealed_to_cpu_batch_async` and
    // `migrate_sealed_to_gpu_batch_async` — the production hot↔warm path
    // used by the substrate persistence thread. They require a CUDA
    // device and a working `kv_migrate_on` kernel; on a CPU-only build
    // each test silently returns early.
    //
    // Byte-identity is checked end-to-end: a known pattern is seeded on
    // GPU, migrated to RAM, then back to VRAM, and the round-tripped
    // bytes must equal the original.

    #[cfg(feature = "cuda")]
    fn cuda_device_or_skip() -> Option<Device> {
        match Device::cuda_if_available(0) {
            Ok(d @ Device::Cuda(_)) => Some(d),
            _ => None,
        }
    }

    /// Build a CUDA-backed `ChunkedKvBacking` with a BF16 float arena.
    #[cfg(feature = "cuda")]
    fn cuda_backing(device: &Device, n_kv_head: usize, head_dim: usize) -> ChunkedKvBacking {
        ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::BF16, device, 256).unwrap()
    }

    /// Seed `n_tokens` of a deterministic BF16 pattern starting at
    /// `pattern_base` and return a sealed snapshot. The pattern is
    /// `bf16::from_f32((pattern_base + i) as f32 * 0.001)` per element,
    /// which gives each element a distinct value so byte-identity
    /// checks catch any chunk-swap or offset bug.
    #[cfg(feature = "cuda")]
    fn seed_and_seal_pattern(
        backing: &ChunkedKvBacking,
        device: &Device,
        n_kv_head: usize,
        head_dim: usize,
        n_tokens: usize,
        pattern_base: u32,
    ) -> SealedSequence {
        use half::bf16;
        let slot = backing.alloc_sequence().unwrap();
        backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
        let total = n_kv_head * n_tokens * head_dim;
        let data: Vec<bf16> = (0..total)
            .map(|i| bf16::from_f32(((pattern_base as usize + i) as f32) * 0.001))
            .collect();
        let k = Tensor::from_vec(
            data.clone(),
            (1, n_kv_head, n_tokens, head_dim),
            &Device::Cpu,
        )
        .unwrap()
        .to_device(device)
        .unwrap();
        let v = k.clone();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, n_tokens);
        backing.record_turn(slot).unwrap()
    }

    /// Walk every unique GID in `sealed` (which must be CPU-resident),
    /// extract the chunk slot's bytes via `read_chunk_into_pinned_bytes`,
    /// and concatenate in iteration order. Used to compare bytes
    /// before/after a round-trip.
    #[cfg(feature = "cuda")]
    fn bytes_of_cpu_sealed(backing: &ChunkedKvBacking, sealed: &SealedSequence) -> Vec<u8> {
        let arena_chunks = arena_gid_stride();
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        backing
            .inner
            .storage
            .read(|s| -> candle::Result<()> {
                for chunk in &sealed.chunks {
                    for gid in chunk.gids.0.iter() {
                        let raw = gid.raw();
                        if !seen.insert(raw) {
                            continue;
                        }
                        let arena_idx = (raw as usize) / arena_chunks;
                        let chunk_idx = (raw as usize) % arena_chunks;
                        let arena = s.arenas().get(&arena_idx).unwrap();
                        let len = chunk_byte_size_of(arena).unwrap();
                        let mut buf = vec![0u8; len];
                        read_chunk_into_pinned_bytes(arena, chunk_idx, &mut buf).unwrap();
                        out.extend(buf);
                    }
                }
                Ok(())
            })
            .unwrap()
            .unwrap();
        out
    }

    /// `migrate_sealed_to_cpu_batch_async` on a single sequence must
    /// produce a CPU-located sealed with the same chunk count, token
    /// count, and per-chunk metadata as the source.
    #[cfg(feature = "cuda")]
    #[test]
    fn batched_async_single_sequence_preserves_structure() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let n_kv_head = 4;
        let head_dim = 32;
        let n_tokens = 40; // 1 full chunk (32) + 1 partial (8)
        let backing = cuda_backing(&device, n_kv_head, head_dim);
        let sealed = seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 0);

        assert_eq!(sealed.token_count, n_tokens);
        assert_eq!(sealed.chunks.len(), 2);
        assert_eq!(sealed.location, ArenaLocation::Gpu);

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        let cpu_batch = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&sealed])
            .expect("batched async migrate must succeed");

        assert_eq!(cpu_batch.len(), 1, "one sequence in → one out");
        let cpu = &cpu_batch[0];
        assert_eq!(cpu.location, ArenaLocation::Cpu);
        assert_eq!(cpu.token_count, n_tokens);
        assert_eq!(cpu.chunks.len(), sealed.chunks.len());
        for (orig, mig) in sealed.chunks.iter().zip(cpu.chunks.iter()) {
            assert_eq!(mig.token_count, orig.token_count);
            assert_eq!(mig.offset, orig.offset);
        }
    }

    /// `migrate_sealed_to_cpu_batch_async` on multiple sequences must
    /// migrate each to its own CPU slots — distinct GIDs across results,
    /// per-sequence structure preserved.
    #[cfg(feature = "cuda")]
    #[test]
    fn batched_async_multiple_sequences_get_distinct_destinations() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let n_kv_head = 2;
        let head_dim = 16;
        let n_tokens = 32; // exactly one full chunk per sequence
        let backing = cuda_backing(&device, n_kv_head, head_dim);

        let s0 = seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 0);
        let s1 = seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 1000);
        let s2 = seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 2000);

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        let migrated = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&s0, &s1, &s2])
            .expect("multi-sequence batched async migrate must succeed");

        assert_eq!(migrated.len(), 3);
        // Every migrated sequence is CPU-located.
        for m in &migrated {
            assert_eq!(m.location, ArenaLocation::Cpu);
            assert_eq!(m.token_count, n_tokens);
            assert_eq!(m.chunks.len(), 1);
        }
        // All destination GIDs across the three sequences must be
        // distinct — no two migrated chunks share a physical CPU slot.
        let mut all_gids: Vec<i64> = Vec::new();
        for m in &migrated {
            for g in m.chunks[0].gids.iter() {
                all_gids.push(g.raw());
            }
        }
        all_gids.sort();
        let unique: std::collections::HashSet<i64> = all_gids.iter().copied().collect();
        assert_eq!(unique.len(), all_gids.len(), "GIDs must be globally unique");
    }

    /// `migrate_sealed_to_gpu_batch_async` on a CPU sequence must produce
    /// a GPU-located sealed with the same per-chunk metadata.
    #[cfg(feature = "cuda")]
    #[test]
    fn batched_async_to_gpu_preserves_structure() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let n_kv_head = 4;
        let head_dim = 32;
        let n_tokens = 40;
        let backing = cuda_backing(&device, n_kv_head, head_dim);
        let gpu_orig = seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 0);

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        // First migrate GPU → CPU, then CPU → GPU.
        let cpu = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_orig])
            .unwrap();
        let gpu_back = backing
            .migrate_sealed_to_gpu_batch_async(&device, &copy_stream, &mut pinned, &[&cpu[0]])
            .expect("batched async to_gpu must succeed");

        assert_eq!(gpu_back.len(), 1);
        let r = &gpu_back[0];
        assert_eq!(r.location, ArenaLocation::Gpu);
        assert_eq!(r.token_count, n_tokens);
        assert_eq!(r.chunks.len(), gpu_orig.chunks.len());
        for (orig, mig) in gpu_orig.chunks.iter().zip(r.chunks.iter()) {
            assert_eq!(mig.token_count, orig.token_count);
            assert_eq!(mig.offset, orig.offset);
        }
    }

    /// **End-to-end byte identity** through the production hot↔warm
    /// path: a known pattern seeded on GPU, migrated to RAM via
    /// `migrate_sealed_to_cpu_batch_async` (kv_migrate_on gather + DtoH
    /// on the copy stream + pinned scratch → Tensor::from_slice +
    /// slice_set), then back to VRAM via
    /// `migrate_sealed_to_gpu_batch_async` (Tensor::to_vec + HtoD on
    /// the copy stream + kv_migrate_on scatter). After the round-trip,
    /// the CPU bytes of the final form must be byte-identical to the
    /// CPU bytes of the first migration.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_cpu_gpu_round_trip_is_byte_identical() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let n_kv_head = 4;
        let head_dim = 32;
        let n_tokens = 40; // 2 chunks: one full, one partial
        let backing = cuda_backing(&device, n_kv_head, head_dim);
        let gpu_orig =
            seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 12345);

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        // GPU → CPU (the bytes the production hot→warm path produces).
        let cpu_first = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_orig])
            .unwrap();
        let first_bytes = bytes_of_cpu_sealed(&backing, &cpu_first[0]);
        assert!(
            !first_bytes.is_empty(),
            "CPU sealed must have non-empty bytes"
        );

        // CPU → GPU (the production warm→hot path).
        let gpu_back = backing
            .migrate_sealed_to_gpu_batch_async(&device, &copy_stream, &mut pinned, &[&cpu_first[0]])
            .unwrap();

        // GPU → CPU again so we can read bytes to compare.
        let cpu_second = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_back[0]])
            .unwrap();
        let second_bytes = bytes_of_cpu_sealed(&backing, &cpu_second[0]);

        assert_eq!(
            first_bytes.len(),
            second_bytes.len(),
            "round-trip must preserve byte length"
        );
        assert_eq!(
            first_bytes, second_bytes,
            "round-trip must be byte-identical — any difference means a chunk got swapped, \
             dropped, or had its bytes corrupted by the gather/scatter pipeline"
        );
    }

    /// **F16 end-to-end byte identity.** Same shape as the BF16
    /// round-trip but with `DType::F16` backing — exercises the
    /// `Tensor::from_slice` + `slice_set` write path with `half::f16`
    /// element casts on the way back to CPU arena slots.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_cpu_gpu_round_trip_f16_is_byte_identical() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use half::f16;
        use std::sync::Arc;

        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let n_kv_head = 4;
        let head_dim = 32;
        let n_tokens = 40; // two chunks: one full, one partial
        let backing =
            ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::F16, &device, 256).unwrap();

        // F16 seed pattern — distinct per-element value, deterministic.
        let slot = backing.alloc_sequence().unwrap();
        backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
        let total = n_kv_head * n_tokens * head_dim;
        let data: Vec<f16> = (0..total)
            .map(|i| f16::from_f32(((12345 + i) as f32) * 0.001))
            .collect();
        let k = Tensor::from_vec(data, (1, n_kv_head, n_tokens, head_dim), &Device::Cpu)
            .unwrap()
            .to_device(&device)
            .unwrap();
        let v = k.clone();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, n_tokens);
        let gpu_orig = backing.record_turn(slot).unwrap();

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        let cpu_first = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_orig])
            .unwrap();
        let first_bytes = bytes_of_cpu_sealed(&backing, &cpu_first[0]);
        assert!(!first_bytes.is_empty());

        let gpu_back = backing
            .migrate_sealed_to_gpu_batch_async(&device, &copy_stream, &mut pinned, &[&cpu_first[0]])
            .unwrap();
        let cpu_second = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_back[0]])
            .unwrap();
        let second_bytes = bytes_of_cpu_sealed(&backing, &cpu_second[0]);

        assert_eq!(first_bytes.len(), second_bytes.len());
        assert_eq!(
            first_bytes, second_bytes,
            "F16 round-trip must be byte-identical"
        );
    }

    /// **R16 end-to-end byte identity.** R16 is the production-default
    /// KV format on Qwen3 (per `CLAUDE.md` — "R16 throughout"). It's
    /// classified as `KvFormat::Quantized(QuantFormat::R16)` in the
    /// type system, so it goes through the same `QTensor::write_bytes_at`
    /// / `read_bytes_at` route as Q8_0 — but with a much simpler
    /// internal byte layout (essentially `f16` storage in
    /// token-oriented `[D, T]` order).
    ///
    /// Seeds via an F16 backing (R16's storage element type), migrates
    /// each chunk to a GPU R16 arena via the existing per-chunk
    /// `migrate_chunk` (which routes through `quantize_into` for
    /// non-fused formats), then runs the batched-async round-trip.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_cpu_gpu_round_trip_r16_is_byte_identical() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use half::f16;
        use std::sync::Arc;

        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        // R16's storage element is F16, and the transpose-then-quantize
        // path in `convert_chunk_data_static` expects matching dtypes.
        // head_dim = 128 → sub_head_dim = 32, n_tokens = 64 → two full
        // chunks (no partial-block padding to reason about).
        let n_kv_head = 2;
        let head_dim = 128;
        let n_tokens = 64;
        let backing =
            ChunkedKvBacking::new(4, n_kv_head, head_dim, DType::F16, &device, 256).unwrap();

        let slot = backing.alloc_sequence().unwrap();
        backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
        let total = n_kv_head * n_tokens * head_dim;
        let data: Vec<f16> = (0..total)
            .map(|i| f16::from_f32(((54321 + i) as f32) * 0.0001))
            .collect();
        let k = Tensor::from_vec(data, (1, n_kv_head, n_tokens, head_dim), &Device::Cpu)
            .unwrap()
            .to_device(&device)
            .unwrap();
        let v = k.clone();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, n_tokens);
        let float_sealed = backing.record_turn(slot).unwrap();

        // Re-route every chunk's GIDs into a GPU R16 arena. R16 isn't
        // in the fused `can_fuse` set in `convert_chunk_data_static`,
        // so this goes through `transpose → quantize_into` — the same
        // path the production code uses for R16 (modulo the production
        // path going through `write_raw_sealed_chunk` rather than
        // `migrate_chunk`).
        let gpu_r16_key = ArenaKey::uniform(
            crate::kv_cache::KvFormat::Quantized(crate::kv_cache::QuantFormat::R16),
            ArenaLocation::Gpu,
        );
        let mut r16_chunks: Vec<SealedChunk> = Vec::with_capacity(float_sealed.chunks.len());
        for chunk in &float_sealed.chunks {
            let new_gids = chunk
                .gids
                .map_unique(|gid| backing.migrate_chunk(gid.raw(), gpu_r16_key.clone()))
                .expect("F16 → GPU R16 migrate must succeed");
            r16_chunks.push(SealedChunk {
                gids: new_gids,
                // Re-placed bytes → drop the resident record (scratch fallback).
                meta: None,
                ..chunk.clone()
            });
        }
        let r16_sealed = SealedSequence {
            chunks: r16_chunks,
            token_count: float_sealed.token_count,
            chunk_size: float_sealed.chunk_size,
            location: ArenaLocation::Gpu,
        };

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        // GPU R16 → CPU R16 (exercises QTensor::write_bytes_at).
        let cpu_first = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&r16_sealed])
            .unwrap();
        let first_bytes = bytes_of_cpu_sealed(&backing, &cpu_first[0]);
        assert!(!first_bytes.is_empty(), "CPU R16 sealed must have bytes");
        assert_eq!(cpu_first[0].location, ArenaLocation::Cpu);

        // CPU R16 → GPU R16 (exercises QTensor::read_bytes_at).
        let gpu_back = backing
            .migrate_sealed_to_gpu_batch_async(&device, &copy_stream, &mut pinned, &[&cpu_first[0]])
            .unwrap();
        assert_eq!(gpu_back[0].location, ArenaLocation::Gpu);

        // GPU R16 → CPU R16 again, compare bytes.
        let cpu_second = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_back[0]])
            .unwrap();
        let second_bytes = bytes_of_cpu_sealed(&backing, &cpu_second[0]);

        assert_eq!(
            first_bytes.len(),
            second_bytes.len(),
            "R16 round-trip must preserve byte length"
        );
        assert_eq!(
            first_bytes, second_bytes,
            "R16 round-trip must be byte-identical — this is the production hot↔warm \
             path's exact byte-level guarantee"
        );
    }

    /// **Quantized end-to-end byte identity.** Same shape as
    /// `gpu_cpu_gpu_round_trip_is_byte_identical` but the SealedSequence
    /// points into a GPU **Q8_0** arena instead of a Float arena, so
    /// the batched-async path goes through `write_bytes_at` /
    /// `read_bytes_at` on `QTensor` rather than through the
    /// `Tensor::from_slice` + `slice_set` route.
    ///
    /// Seeds Float bytes, migrates each chunk to a GPU Q8_0 arena via
    /// `migrate_chunk` (the existing per-chunk fused-quantize kernel),
    /// then runs the batched-async migrate GPU→CPU→GPU→CPU. The two
    /// CPU byte vectors must be identical — any difference means the
    /// Quantized branch of `write_chunk_from_pinned_bytes` or
    /// `read_chunk_into_pinned_bytes` corrupted, swapped, or dropped a
    /// chunk.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_cpu_gpu_round_trip_quantized_is_byte_identical() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        // head_dim = 128 → sub_head_dim = 32 → matches the Q8_0
        // fused-quantize fast path in convert_chunk_data_static. Two
        // full chunks (n_tokens = 64) keeps every chunk's element
        // count a multiple of Q8_0's block_size (32).
        let n_kv_head = 2;
        let head_dim = 128;
        let n_tokens = 64;
        let backing = cuda_backing(&device, n_kv_head, head_dim);
        let float_sealed =
            seed_and_seal_pattern(&backing, &device, n_kv_head, head_dim, n_tokens, 4242);

        // Re-route every chunk's GIDs through a GPU Q8_0 arena via
        // the existing fused Float→Quant migrate_chunk path.
        let gpu_q8_key = ArenaKey::uniform(
            crate::kv_cache::KvFormat::Quantized(crate::kv_cache::QuantFormat::Q8_0),
            ArenaLocation::Gpu,
        );
        let mut quant_chunks: Vec<SealedChunk> = Vec::with_capacity(float_sealed.chunks.len());
        for chunk in &float_sealed.chunks {
            let new_gids = chunk
                .gids
                .map_unique(|gid| backing.migrate_chunk(gid.raw(), gpu_q8_key.clone()))
                .expect("Float → GPU Q8_0 migrate must succeed for fused path");
            quant_chunks.push(SealedChunk {
                gids: new_gids,
                // Re-placed bytes → drop the resident record (scratch fallback).
                meta: None,
                ..chunk.clone()
            });
        }
        let quant_sealed = SealedSequence {
            chunks: quant_chunks,
            token_count: float_sealed.token_count,
            chunk_size: float_sealed.chunk_size,
            location: ArenaLocation::Gpu,
        };

        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        // GPU Q8_0 → CPU Q8_0 (write_bytes_at path on the dest side).
        let cpu_first = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&quant_sealed])
            .unwrap();
        let first_bytes = bytes_of_cpu_sealed(&backing, &cpu_first[0]);
        assert!(!first_bytes.is_empty(), "CPU Q8_0 sealed must have bytes");
        assert_eq!(cpu_first[0].location, ArenaLocation::Cpu);

        // CPU Q8_0 → GPU Q8_0 (read_bytes_at path on the source side).
        let gpu_back = backing
            .migrate_sealed_to_gpu_batch_async(&device, &copy_stream, &mut pinned, &[&cpu_first[0]])
            .unwrap();
        assert_eq!(gpu_back[0].location, ArenaLocation::Gpu);

        // GPU Q8_0 → CPU Q8_0 again so we can compare bytes.
        let cpu_second = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&gpu_back[0]])
            .unwrap();
        let second_bytes = bytes_of_cpu_sealed(&backing, &cpu_second[0]);

        assert_eq!(
            first_bytes.len(),
            second_bytes.len(),
            "Q8_0 round-trip must preserve byte length"
        );
        assert_eq!(
            first_bytes, second_bytes,
            "Q8_0 round-trip must be byte-identical — any difference means the \
             Quantized branch of write_bytes_at / read_bytes_at corrupted, swapped, \
             or dropped a chunk"
        );
    }

    /// Empty input to the batched-async paths is a no-op: returns an
    /// empty Vec without touching the device or the pinned scratch.
    #[cfg(feature = "cuda")]
    #[test]
    fn batched_async_empty_input_is_noop() {
        use candle::cuda_backend::cudarc::driver::CudaStream;
        use candle::quantized::pinned_staging::PinnedBuf;
        use std::sync::Arc;

        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let backing = cuda_backing(&device, 2, 16);
        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        let copy_stream: Arc<CudaStream> = cuda_dev.cuda_context().new_stream().unwrap();
        let mut pinned: Option<PinnedBuf> = None;

        let out = backing
            .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[])
            .unwrap();
        assert!(out.is_empty());
        assert!(pinned.is_none(), "no scratch allocation on empty input");

        let out_gpu = backing
            .migrate_sealed_to_gpu_batch_async(&device, &copy_stream, &mut pinned, &[])
            .unwrap();
        assert!(out_gpu.is_empty());
    }
}
