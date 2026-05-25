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
use super::{Arena, ArenaKey, ChunkedKvBacking, CompressionPolicy, StoragePolicy};
// Import from kv_cache module (grandparent)
use crate::kv_cache::arena_table::{ArenaLocation, N_PALETTE};
use crate::kv_cache::KvFormat;
#[cfg(feature = "cuda")]
use crate::kv_cache::QuantFormat;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::SELECT_FMT_F16;
use candle::quantized::pinned_staging::Generation;
use candle::quantized::QTensor;
use candle::{DType, Device, Result, Tensor};

// Shared production thresholds are defined in compression_policy.rs and are
// consumed here directly so the runtime and table harness stay in sync.

#[cfg(feature = "cuda")]
#[inline]
fn needs_reconcile_source_format(format: KvFormat) -> bool {
    matches!(
        format,
        KvFormat::Float(_) | KvFormat::Quantized(QuantFormat::R16)
    )
}

impl ChunkedKvBacking {
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
            // Re-validate source (still must exist and not be tombstoned)
            if source_arena_idx >= s.arena_count() {
                return Ok(false);
            }
            let source_key = s
                .arena_key(source_arena_idx)
                .ok_or_else(|| candle::Error::Msg("source arena not found".into()))?;

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
                // Source was tombstoned between locks ? skip migration.
                // new_gid drops here, returning GID to pool.
                Ok(ChunkGid::detached(source_gid as i64))
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

    /// Reconcile chunks to match a storage policy after decode/prefill.
    ///
    /// This migrates sealed chunks to their target format/location according to
    /// the policy. The active (partial) chunk is NEVER reconciled because:
    /// 1. It's still being written to
    /// 2. Quantized formats require full blocks
    ///
    /// For GPU Float?Quant migrations, uses batched fused transpose+quantize
    /// to process all chunks in a single kernel launch for efficiency.
    ///
    /// # Arguments
    /// * `batch_idx` - Sequence slot index
    /// * `seq_len` - Current sequence length (to determine which chunks are sealed)
    /// * `policy` - Target storage policy for sealed chunks
    ///
    /// # Returns
    /// Number of chunks migrated.
    /// Batch-optimized reconcile across multiple sequences in one lock acquisition.
    ///
    /// Instead of calling `reconcile()` per-sequence (120 lock acquisitions),
    /// this takes a single state lock and a single storage read lock to check
    /// all sealed chunks across all sequences. This eliminates the per-sequence
    /// lock overhead that dominated decode time.
    ///
    /// Returns total number of chunks migrated across all sequences.
    /// Reconcile all sealed chunks within a lookback window across a batch.
    ///
    /// For each sequence, only blocks whose last token falls within
    /// `[seq_len - lookback_tokens, seq_len)` are considered.  Blocks outside
    /// that window were either already reconciled in a prior call or belong to
    /// a different pass and should not be touched.
    ///
    /// - `lookback_tokens = 1`         — decode fast-path: only fires when seq_len
    ///   is an exact multiple of CHUNK_SIZE (31 of 32 steps are O(1) no-ops).
    /// - `lookback_tokens = prefill_n` — prefill path: covers all blocks sealed by
    ///   the current prefill batch.
    /// - `lookback_tokens = usize::MAX` — full scan: reconcile every sealed block.
    ///
    /// Returns the total number of chunks migrated.
    /// Collect GPU float→quant migrations for the given batch window.
    ///
    /// Returns `Some((migrations, target_k_key, target_v_key))` when there is work
    /// to do, `None` when all blocks are already at the target format or the window
    /// is empty.  The caller is responsible for enqueueing and joining.
    #[cfg(feature = "cuda")]
    fn collect_gpu_migrations(
        &self,
        batch_indices: &[(usize, usize)],
        policy: StoragePolicy,
        compression: Option<&CompressionPolicy>,
        lookback_tokens: usize,
    ) -> Result<Option<Vec<super::bg_quantizer::ChunkMigration>>> {
        if lookback_tokens == 0 || batch_indices.is_empty() {
            tracing::debug!(
                "bg_quantizer: collect_gpu_migrations early-out (lookback={lookback_tokens} batch_indices_len={})",
                batch_indices.len()
            );
            return Ok(None);
        }
        let any_eligible = batch_indices.iter().any(|&(_, seq_len)| {
            let n_sealed = seq_len / CHUNK_SIZE;
            let start_blk = seq_len.saturating_sub(lookback_tokens) / CHUNK_SIZE;
            n_sealed > 0 && start_blk < n_sealed
        });
        if self.layer_idx == 0 {
            tracing::debug!(
                "bg_quantizer: collect_gpu_migrations layer=0 lookback={lookback_tokens} batch_indices={:?} any_eligible={any_eligible}",
                batch_indices
            );
        }
        if !any_eligible {
            return Ok(None);
        }

        let target_key = policy.to_arena_key();
        let v_target_key_val = self.inner.storage.default_v_key();
        let v_is_float = matches!(v_target_key_val.format, KvFormat::Float(_));

        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let layer_idx = state.layer_idx;
        let adaptive = compression.is_some();
        let mut gpu_float_to_quant: Vec<super::bg_quantizer::ChunkMigration> = Vec::new();

        self.inner.storage.read(|s| {
            for &(batch_idx, seq_len) in batch_indices {
                let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                    Some(s) => s,
                    None => continue,
                };
                if slot.block_count() == 0 {
                    continue;
                }
                let n_sealed = seq_len / CHUNK_SIZE;
                let start_blk = seq_len.saturating_sub(lookback_tokens) / CHUNK_SIZE;
                let limit = n_sealed.min(slot.block_count()).min(state.max_blocks);
                if start_blk >= limit {
                    continue;
                }
                for blk in (start_blk..limit).rev() {
                    let cw = match slot.chunk_at(blk) {
                        Some(cw) => cw,
                        _ => continue,
                    };
                    let first_k_ai = cw.gids.k_gid(0).arena_idx();
                    if let Some(k) = s.arena_key(first_k_ai) {
                        if k == target_key {
                            let first_v_ai = cw.gids.v_gid(0).arena_idx();
                            if let Some(vk) = s.arena_key(first_v_ai) {
                                if vk == v_target_key_val {
                                    break;
                                }
                            }
                        }
                    }

                    let k_arena_indices = cw.gids.unique_k_arena_indices();
                    let v_arena_indices = cw.gids.unique_v_arena_indices();

                    let k_has_pending_src = k_arena_indices.iter().any(|&ai| {
                        s.arena_key(ai)
                                .map(|k| needs_reconcile_source_format(k.format))
                                .unwrap_or(false)
                    });
                    let v_has_pending_src = v_arena_indices.iter().any(|&ai| {
                        s.arena_key(ai)
                                .map(|k| needs_reconcile_source_format(k.format))
                                .unwrap_or(false)
                    });
                    if !k_has_pending_src && !v_has_pending_src {
                        break;
                    }

                    let gids = cw.gids.clone();

                    let k_needs = k_has_pending_src
                        && k_arena_indices.iter().any(|&ai| {
                            s.arena_key(ai).map(|k| k != target_key).unwrap_or(false)
                        });
                    let v_needs = v_has_pending_src
                        && v_arena_indices.iter().any(|&ai| {
                            s.arena_key(ai)
                                    .map(|k| k != v_target_key_val)
                                    .unwrap_or(false)
                        });
                    if !k_needs && !v_needs {
                        continue;
                    }

                    let k_all_gpu_float = k_needs
                        && gids.unique_k_arena_indices().iter().all(|&ai| {
                            s.arena_key(ai)
                                    .map(|k| {
                                        matches!(k.location, ArenaLocation::Gpu)
                                            && matches!(
                                                k.format,
                                                KvFormat::Float(_)
                                                    | KvFormat::Quantized(QuantFormat::R16)
                                            )
                                    })
                                    .unwrap_or(false)
                        });
                    let v_all_gpu_float = v_needs
                        && gids.unique_v_arena_indices().iter().all(|&ai| {
                            s.arena_key(ai)
                                    .map(|k| {
                                        matches!(k.location, ArenaLocation::Gpu)
                                            && matches!(
                                                k.format,
                                                KvFormat::Float(_)
                                                    | KvFormat::Quantized(QuantFormat::R16)
                                            )
                                    })
                                    .unwrap_or(false)
                        });
                    if k_all_gpu_float
                        && matches!(target_key.location, ArenaLocation::Gpu)
                        && matches!(target_key.format, KvFormat::Quantized(_))
                        && (!v_needs || v_is_float || v_all_gpu_float)
                    {
                        gpu_float_to_quant.push(super::bg_quantizer::ChunkMigration {
                            layer_idx,
                            batch_idx,
                            block_idx: blk,
                            head_gids: gids,
                        });
                        continue;
                    }

                    if adaptive {
                        let all_already_quant = gids.unique_k_arena_indices().iter().all(|&ai| {
                            s.arena_key(ai)
                                    .map(|k| matches!(k.format, KvFormat::Quantized(_)))
                                    .unwrap_or(false)
                        });
                        if all_already_quant
                            && matches!(target_key.format, KvFormat::Quantized(_))
                        {
                            continue;
                        }
                    }
                    candle::bail!(
                        "reconcile_multi: encountered non-fast-path migration at batch_idx={} blk={} (slow path removed)",
                        batch_idx,
                        blk
                    );
                }
            }
            Ok::<(), candle::Error>(())
        })??;

        if gpu_float_to_quant.is_empty() {
            return Ok(None);
        }
        Ok(Some(gpu_float_to_quant))
    }

    pub fn reconcile_multi(
        &self,
        batch_indices: &[(usize, usize)], // (batch_idx, seq_len)
        policy: StoragePolicy,
        compression: Option<&CompressionPolicy>,
        lookback_tokens: usize,
        generation: &Generation,
    ) -> Result<usize> {
        let _ = &generation;
        #[cfg(feature = "cuda")]
        if let Some(migrations) =
            self.collect_gpu_migrations(batch_indices, policy, compression, lookback_tokens)?
        {
            let count = migrations.len();
            self.enqueue_reconcile_batch(migrations, generation)?;
            return Ok(count);
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (batch_indices, policy, compression, lookback_tokens);
        Ok(0)
    }

    /// Collect this layer's migrations into a WorkItem ready for enqueueing.
    /// Returns `None` when there is nothing to do.
    #[cfg(feature = "cuda")]
    pub(crate) fn collect_work_item(
        &self,
        batch_indices: &[(usize, usize)],
        policy: StoragePolicy,
        compression: Option<&CompressionPolicy>,
        lookback_tokens: usize,
    ) -> Result<Option<super::bg_quantizer::WorkItem>> {
        use super::bg_quantizer::WorkItem;
        if let Some(migrations) =
            self.collect_gpu_migrations(batch_indices, policy, compression, lookback_tokens)?
        {
            return Ok(Some(WorkItem { migrations }));
        }
        Ok(None)
    }

    /// Collect migrations from every layer, push them all onto the shared background
    /// quantizer in one lock acquisition, then join once.
    ///
    /// One GPU pass covering all N_layers × N_sequences blocks, with a single
    /// bg-thread round-trip regardless of the number of layers.
    pub(crate) fn reconcile_all_layers(
        &self,
        layers: &[(
            &ChunkedKvBacking,
            StoragePolicy,
            Option<&CompressionPolicy>,
            &[(usize, usize)], // (batch_idx, seq_len)
        )],
        lookback_tokens: usize,
    ) -> Result<usize> {
        if layers.is_empty() {
            return Ok(0);
        }
        #[cfg(feature = "cuda")]
        {
            use crate::kv_cache::chunked::bg_quantizer::WorkItem;

            let mut work_items: Vec<WorkItem> = Vec::new();
            let mut total = 0usize;

            for &(backing, policy, compression, batch_indices) in layers {
                if let Some(item) = backing.collect_work_item(
                    batch_indices,
                    policy,
                    compression,
                    lookback_tokens,
                )? {
                    total += item.migrations.len();
                    work_items.push(item);
                }
            }

            tracing::debug!(
                "bg_quantizer: reconcile_all_layers layers={} work_items={} total_migrations={}",
                layers.len(),
                work_items.len(),
                total
            );
            if !work_items.is_empty() {
                self.bg_quantizer.enqueue_work_items_batch(work_items);
                //self.bg_quantizer.join();
            }
            return Ok(total);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = lookback_tokens;
            Ok(0)
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
        let gids = self.alloc_sealed_block(
            batch_idx, block_idx, k_format, v_format, k_pal, v_pal, k_scale, v_scale,
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
        k_format: KvFormat,
        v_format: KvFormat,
        k_pal: std::sync::Arc<Vec<u8>>,
        v_pal: std::sync::Arc<Vec<u8>>,
        k_scale: std::sync::Arc<Vec<f32>>,
        v_scale: std::sync::Arc<Vec<f32>>,
    ) -> Result<HeadGids> {
        let n_kv_head = self.inner.n_kv_head;
        let location = self.inner.storage.default_location();
        let target_key = ArenaKey::uniform(k_format, location);
        let value_key = ArenaKey::uniform(v_format, location);

        // Ensure the block table has room for this block index and materialize
        // the destination sequence slot before attaching the fresh GIDs.
        self.ensure_max_blocks(block_idx + 1)?;
        self.ensure_for_offset(batch_idx, block_idx * CHUNK_SIZE, CHUNK_SIZE)?;

        // Allocate GIDS_PER_HEAD * n_kv_head destination chunks.
        let mut gid_vec: Vec<ChunkGid> = Vec::with_capacity(GIDS_PER_HEAD * n_kv_head);
        {
            let _state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for _h in 0..n_kv_head {
                for _p in 0..N_PALETTE {
                    let k_gid = self.alloc_chunk_for_key(target_key.clone())?;
                    let v_gid = self.alloc_chunk_for_key(value_key.clone())?;
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

    /// The `(k_format, v_format)` a sealed chunk's bytes are stored in —
    /// read from the arenas its head-0 K/V GIDs resolve to.
    ///
    /// Adaptive quantization picks K and V formats independently per block,
    /// so persistence must record both. Used by the seal-time gather path.
    pub fn sealed_chunk_kv_formats(
        &self,
        chunk: &SealedChunk,
    ) -> candle::Result<(crate::kv_cache::KvFormat, crate::kv_cache::KvFormat)> {
        let k_arena = chunk.gids.k_gid(0).arena_idx();
        let v_arena = chunk.gids.v_gid(0).arena_idx();
        self.inner.storage.read(|s| {
            let k = s
                .arena_key(k_arena)
                .map(|key| key.format)
                .ok_or_else(|| candle::Error::Msg(format!("k arena {k_arena} not found")))?;
            let v = s
                .arena_key(v_arena)
                .map(|key| key.format)
                .ok_or_else(|| candle::Error::Msg(format!("v arena {v_arena} not found")))?;
            Ok((k, v))
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
        backing.record_turn(slot, n_tokens).unwrap()
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
}
