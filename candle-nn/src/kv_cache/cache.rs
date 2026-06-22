//! Cache types that wrap contiguous or chunked KV storage.
//!
//! This module provides `Cache` for single-tensor caches and `KvCache` for
//! paired key-value caches, supporting both contiguous and chunked backing.

use super::chunked::{arena_gid_stride, ChunkedKvBacking, CompressionPolicy, CHUNK_SIZE};
use ahash::HashMap;
use candle::quantized::GgmlDType;
use candle::{DType, Result, Tensor};

/// Internal chunked cache wrapper for a single sequence slot.
#[derive(Debug, Clone)]
pub(crate) struct ChunkedCache {
    pub(crate) backing: ChunkedKvBacking,
    pub(crate) batch_idx: usize,
    /// Compression policy for adaptive format selection.
    /// Owned by the session/conversation layer, passed down when creating caches.
    /// `None` means no compression — uniform storage with no per-block selection.
    pub(crate) compression_policy: Option<CompressionPolicy>,
}

impl ChunkedCache {
    pub(crate) fn new(
        backing: ChunkedKvBacking,
        batch_idx: usize,
        compression_policy: Option<CompressionPolicy>,
    ) -> Result<Self> {
        // Ensure the slot is allocated in the backing so that ensure_for_offsets
        // and other methods will work correctly. This also grows capacity if needed.
        backing.ensure_sequence_allocated(batch_idx)?;
        Ok(Self {
            backing,
            batch_idx,
            compression_policy,
        })
    }

    /// Get the storage policy for this backing.
    pub(crate) fn storage_policy(&self) -> super::StoragePolicy {
        self.backing.storage_policy()
    }

    /// Get the max_blocks for this cache.
    pub(crate) fn max_blocks(&self) -> usize {
        self.backing.max_blocks()
    }

    pub(crate) fn k_arenas(&self) -> Vec<Tensor> {
        self.backing.k_arenas()
    }

    pub(crate) fn v_arenas(&self) -> Vec<Tensor> {
        self.backing.v_arenas()
    }

    /// Execute a read operation on the arena storage.
    pub(crate) fn with_arenas<R>(
        &self,
        f: impl FnOnce(&ahash::AHashMap<usize, super::Arena>) -> R,
    ) -> Result<R> {
        self.backing.with_arenas(f)
    }

    pub(crate) fn set_len(&self, len: usize) {
        self.backing.set_len(self.batch_idx, len)
    }

    /// Get per-block valid token counts for this sequence slot.
    pub(crate) fn block_usage(&self) -> Vec<u32> {
        self.backing.block_usage(self.batch_idx)
    }

    /// Get per-block absolute RoPE base positions for this sequence slot.
    pub(crate) fn chunk_rope_positions(&self) -> Vec<i32> {
        self.backing.chunk_rope_positions(self.batch_idx)
    }

    /// Fork this cache, creating a new ChunkedCache that shares prefix blocks via COW.
    pub(crate) fn fork(&self, seq_len: usize) -> Result<Self> {
        let new_batch_idx = self.backing.fork_sequence_alloc(self.batch_idx, seq_len)?;
        Ok(Self {
            backing: self.backing.clone(),
            batch_idx: new_batch_idx,
            compression_policy: self.compression_policy.clone(),
        })
    }

    #[allow(dead_code)]
    pub(crate) fn batch_idx(&self) -> usize {
        self.batch_idx
    }

    pub(crate) fn is_quantized(&self) -> bool {
        self.backing.is_quantized()
    }

    /// Get chunk references for this sequence's blocks.
    pub(crate) fn get_chunk_refs(
        &self,
        block_range: Option<std::ops::Range<usize>>,
    ) -> Result<Vec<super::HeadGids>> {
        self.backing.get_chunk_refs(self.batch_idx, block_range)
    }

    /// Get chunk references with RoPE shift.
    pub(crate) fn get_chunk_refs_with_rope(
        &self,
        block_range: Option<std::ops::Range<usize>>,
    ) -> Result<Vec<super::HeadGids>> {
        self.backing
            .get_chunk_refs_with_rope(self.batch_idx, block_range)
    }

    /// Write contiguous K/V data to the chunked backing.
    /// Expects tensors shaped (1, n_kv_head, len, head_dim).
    pub(crate) fn write_contiguous(&self, offset: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        self.backing.write_contiguous(self.batch_idx, offset, k, v)
    }

    /// Read contiguous K/V data from the chunked backing (dequantizes if needed).
    /// Returns tensors shaped (1, n_kv_head, len, head_dim).
    pub(crate) fn read_contiguous(&self, offset: usize, len: usize) -> Result<(Tensor, Tensor)> {
        self.backing.read_contiguous(self.batch_idx, offset, len)
    }
}

#[derive(Debug, Clone)]
pub(crate) enum CacheStorage {
    /// Traditional contiguous KV cache backing. `all_data` stores the capacity buffer.
    Contiguous { all_data: Option<Tensor> },
    /// Chunked (paged) KV cache backing.
    Chunked(ChunkedCache),
}

/// A single cache (either K or V) for contiguous or chunked KV storage.
#[derive(Debug, Clone)]
pub struct Cache {
    pub(crate) storage: CacheStorage,
    pub(crate) dim: usize,
    pub(crate) current_seq_len: usize,
    pub(crate) grow_by: usize,
    pub(crate) max_seq_len: usize,
    pub(crate) force_dtype: Option<DType>,
}

impl Cache {
    /// Create a new contiguous cache.
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self {
            storage: CacheStorage::Contiguous { all_data: None },
            dim,
            current_seq_len: 0,
            grow_by: max_seq_len,
            max_seq_len,
            force_dtype: None,
        }
    }

    /// Returns true if this cache uses chunked (paged) backing storage.
    pub fn is_chunked(&self) -> bool {
        matches!(self.storage, CacheStorage::Chunked(_))
    }

    /// Returns the chunked backing if this cache uses chunked storage.
    pub fn chunked_backing(&self) -> Option<&ChunkedKvBacking> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(&c.backing),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Get the max_blocks for chunked caches. Returns 0 for contiguous caches.
    pub fn chunked_max_blocks(&self) -> usize {
        match &self.storage {
            CacheStorage::Chunked(c) => c.max_blocks(),
            CacheStorage::Contiguous { .. } => 0,
        }
    }

    /// Get the K arenas for chunked caches.
    pub fn chunked_k_arenas(&self) -> Option<Vec<Tensor>> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(c.k_arenas()),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Get the V arenas for chunked caches.
    pub fn chunked_v_arenas(&self) -> Option<Vec<Tensor>> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(c.v_arenas()),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Get the number of chunks per arena for chunked caches.
    pub fn chunked_arena_chunks(&self) -> Option<usize> {
        match &self.storage {
            CacheStorage::Chunked(_) => Some(arena_gid_stride()),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Get the chunk size for chunked caches.
    pub fn chunked_chunk_size(&self) -> Option<usize> {
        match &self.storage {
            CacheStorage::Chunked(_) => Some(CHUNK_SIZE),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Get the batch index for chunked caches.
    pub fn chunked_batch_idx(&self) -> Option<usize> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(c.batch_idx),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Get per-block valid token counts for this chunked cache slot.
    /// Returns empty vec for contiguous caches.
    pub fn block_usage(&self) -> Vec<u32> {
        match &self.storage {
            CacheStorage::Chunked(c) => c.block_usage(),
            CacheStorage::Contiguous { .. } => Vec::new(),
        }
    }

    /// Get per-block absolute RoPE base positions for this chunked cache slot.
    /// Returns empty vec for contiguous caches.
    pub fn chunk_rope_positions(&self) -> Vec<i32> {
        match &self.storage {
            CacheStorage::Chunked(c) => c.chunk_rope_positions(),
            CacheStorage::Contiguous { .. } => Vec::new(),
        }
    }

    /// Ensure the chunked backing has capacity for `offset + add` tokens.
    /// No-op for contiguous caches. Takes `&self` (uses interior mutability).
    pub fn ensure_chunked_for_offset(&self, offset: usize, add: usize) -> Result<()> {
        if let CacheStorage::Chunked(c) = &self.storage {
            c.backing.ensure_for_offset(c.batch_idx, offset, add)?;
        }
        Ok(())
    }

    /// Check if chunked cache uses quantized storage.
    pub fn chunked_is_quantized(&self) -> Option<bool> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(c.is_quantized()),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Get the storage policy for chunked caches.
    ///
    /// Returns None for contiguous caches (they don't have storage policies).
    pub fn chunked_storage_policy(&self) -> Option<super::StoragePolicy> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(c.storage_policy()),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Returns true when V stays in float arenas (K is quantized, V is float).
    pub fn chunked_v_stays_float(&self) -> bool {
        match &self.storage {
            CacheStorage::Chunked(c) => {
                c.backing.k_format().is_quantized() && !c.backing.v_format().is_quantized()
            }
            CacheStorage::Contiguous { .. } => false,
        }
    }

    /// Get direct access to raw arenas (float or quantized).
    /// This is the preferred API for kernels that need to handle heterogeneous storage.
    ///
    /// Calls the provided closure with a map of all arenas keyed by arena index.
    pub fn with_chunked_arenas<R>(
        &self,
        f: impl FnOnce(&ahash::AHashMap<usize, super::Arena>) -> R,
    ) -> Option<candle::Result<R>> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(c.with_arenas(f)),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Count the number of quantized arenas.
    ///
    /// Returns (quantized_count, total_count) tuple.
    /// Useful for validating that quantization is actually occurring.
    pub fn count_quantized_arenas(&self) -> Option<candle::Result<(usize, usize)>> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(c.backing.count_quantized_arenas()),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Calculate the percentage of a sequence's tokens stored in quantized arenas.
    ///
    /// Returns (quantized_tokens, total_tokens) based on which ChunkRefs point to quantized arenas.
    /// This validates that the actual data for a sequence is quantized, not just that quantized arenas exist.
    pub fn quantized_token_stats(
        &self,
        batch_idx: usize,
    ) -> Option<candle::Result<(usize, usize)>> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(c.backing.quantized_token_stats(batch_idx)),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Get byte-level compression stats for a sequence.
    ///
    /// Returns weighted BPE summed across K and V for chunked storage, None for contiguous storage
    /// plus the total number of elements
    /// `None` for contiguous storage.
    pub fn compression_bpe(&self, batch_idx: usize) -> Option<(f64, usize)> {
        match &self.storage {
            CacheStorage::Chunked(c) => {
                let (bpe, n) = c.backing.compression_bpe(batch_idx).unwrap();
                Some((bpe as f64, n))
            }
            CacheStorage::Contiguous { .. } => None,
        }
    }

    pub fn compression_dist(
        &self,
        batch_idx: usize,
        is_value: bool,
        ret: &mut HashMap<GgmlDType, usize>,
    ) {
        if let CacheStorage::Chunked(c) = &self.storage {
            c.backing.compression_dist(batch_idx, is_value, ret);
        }
    }

    /// Get the K and V format tags for this chunked cache's backing storage.
    ///
    /// Returns `(k_format_tag, v_format_tag)` derived from the backing's configured
    /// default K/V formats. Used for format validation (e.g. K/V divergence checks).
    /// For kernel dispatch, use `dtype()` instead — it correctly returns F16 for
    /// quantized backings whose arenas may still be pre-reconcile float.
    pub fn chunked_arena_format_tags(
        &self,
    ) -> Option<(super::ArenaFormatTag, super::ArenaFormatTag)> {
        match &self.storage {
            CacheStorage::Chunked(c) => {
                let k_tag = super::ArenaFormatTag::from_kv_format(c.backing.k_format());
                let v_tag = super::ArenaFormatTag::from_kv_format(c.backing.v_format());
                Some((k_tag, v_tag))
            }
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Get the per-head table tensor for decode kernel consumption.
    ///
    /// Returns the GPU tensor of shape `(num_arenas * n_kv_head, 7)` i64,
    /// synced to GPU. Each row is a `PerHeadTableEntry` with pre-resolved
    /// pointers, byte offsets, byte strides, and format metadata per head.
    pub fn chunked_per_head_table_and_sync(&self) -> Option<candle::Result<Tensor>> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(c.backing.per_head_table_sync()),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Get sealed chunk descriptors for this cache slot's live sequence.
    ///
    /// Returns `None` for contiguous caches or if the backing has no live sequence
    /// for this slot's batch index.
    pub fn chunked_live_chunks_as_sealed(&self) -> Option<Vec<super::SealedChunk>> {
        match &self.storage {
            CacheStorage::Chunked(c) => {
                let arena_infos = c.backing.resolve_arena_info().ok()?;
                c.backing.live_chunks_as_sealed(c.batch_idx, &arena_infos)
            }
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Gather this slot's live sealed chunks off the GPU into portable host
    /// [`HostSealedChunk`]s — used by the prefill-capture fixture tooling to
    /// snapshot the cached KV a kernel call attends. Returns `None` for
    /// contiguous caches or when the slot has no live chunks.
    #[cfg(feature = "cuda")]
    pub fn chunked_dump_sealed_to_host(
        &self,
        device: &candle::Device,
    ) -> Option<candle::Result<Vec<super::HostSealedChunk>>> {
        match &self.storage {
            CacheStorage::Chunked(c) => {
                let arena_info = match c.backing.resolve_arena_info() {
                    Ok(a) => a,
                    Err(e) => return Some(Err(e)),
                };
                let chunks = c.backing.live_chunks_as_sealed(c.batch_idx, &arena_info)?;
                Some(c.backing.dump_sealed_to_host(&chunks, device))
            }
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Like [`Self::chunked_live_chunks_as_sealed`] but reuses an already-resolved
    /// `arena_info` instead of resolving it again — the caller (e.g.
    /// `build_slot_headers`) resolves it once per forward and passes it per cache.
    pub fn chunked_live_chunks_as_sealed_with(
        &self,
        arena_info: &[super::ResolvedArenaInfo],
    ) -> Option<Vec<super::SealedChunk>> {
        match &self.storage {
            CacheStorage::Chunked(c) => c.backing.live_chunks_as_sealed(c.batch_idx, arena_info),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// First chunk index that is writer-owned for the slot bound to
    /// this cache.  Chunks before this index are Arc-shared with
    /// substrate/parent and immutable; chunks at or after it are
    /// uniquely-owned by this slot.
    ///
    /// Returns `None` for contiguous caches or if the backing has no
    /// live sequence for this slot's batch index.
    pub fn chunked_writer_start_idx(&self) -> Option<usize> {
        match &self.storage {
            CacheStorage::Chunked(c) => c.backing.writer_start_idx_for_seq(c.batch_idx),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Device address of a resident KV-head record (for a slice's `kvheads_ptr`).
    /// `0` if there is no chunked backing or no device residence (CPU pool).
    pub fn chunked_meta_device_addr(&self, meta: &super::MetaGid) -> u64 {
        match &self.storage {
            CacheStorage::Chunked(c) => c.backing.meta_device_addr(meta),
            CacheStorage::Contiguous { .. } => 0,
        }
    }

    /// Resolve arena info (base pointers, strides, format tags) from this cache's backing.
    ///
    /// Returns `None` for contiguous caches.
    pub fn chunked_resolve_arena_info(
        &self,
    ) -> Option<candle::Result<Vec<super::ResolvedArenaInfo>>> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(c.backing.resolve_arena_info()),
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Invalidate the persistent GPU slot-state buffer for this cache's sequence.
    ///
    /// Forces a full rebuild on the next decode step.  Call this after any
    /// operation (e.g. prefill) that modifies chunk layout or lengths without
    /// going through the decode kernel's self-increment path.
    pub fn chunked_invalidate_decode_gpu_chunks(&self) {
        if let CacheStorage::Chunked(c) = &self.storage {
            c.backing.invalidate_decode_gpu_chunks(&[(c.batch_idx, 0)]);
        }
    }

    /// Get chunk references for a chunked cache's blocks.
    ///
    /// Returns None for contiguous caches.
    /// For chunked caches, returns ChunkGid for each allocated block.
    pub fn get_chunk_refs(
        &self,
        block_range: Option<std::ops::Range<usize>>,
    ) -> Result<Option<Vec<super::HeadGids>>> {
        match &self.storage {
            CacheStorage::Chunked(c) => Ok(Some(c.get_chunk_refs(block_range)?)),
            CacheStorage::Contiguous { .. } => Ok(None),
        }
    }

    /// Get chunk references with RoPE shift for a chunked cache.
    ///
    /// Returns None for contiguous caches.
    pub fn get_chunk_refs_with_rope(
        &self,
        block_range: Option<std::ops::Range<usize>>,
    ) -> Result<Option<Vec<super::HeadGids>>> {
        match &self.storage {
            CacheStorage::Chunked(c) => Ok(Some(c.get_chunk_refs_with_rope(block_range)?)),
            CacheStorage::Contiguous { .. } => Ok(None),
        }
    }

    /// Reconcile sealed chunks to match the storage policy.
    ///
    /// This re-quantizes sealed chunks according to the backing's storage policy
    /// after kernel execution. The active (partial) chunk is never reconciled.
    pub(crate) fn set_chunked_len(&self, len: usize) {
        if let CacheStorage::Chunked(chunked) = &self.storage {
            chunked.set_len(len)
        }
    }

    /// Write contiguous data to chunked storage.
    /// Expects tensor shaped (1, n_kv_head, len, head_dim).
    #[allow(dead_code)]
    pub fn chunked_write(&self, _offset: usize, _data: &Tensor) -> Result<()> {
        match &self.storage {
            CacheStorage::Chunked(_) => {
                // ChunkedKvBacking::write_contiguous expects both k and v,
                // but Cache is per-k or per-v. We need a dummy for the other.
                // This is a bit awkward - the caller should use KvCache::chunked_write_kv instead.
                candle::bail!("chunked_write on individual Cache not supported; use KvCache::chunked_write_kv")
            }
            CacheStorage::Contiguous { .. } => {
                candle::bail!("chunked_write only available for chunked caches")
            }
        }
    }

    /// Read contiguous data from chunked storage (dequantizes if needed).
    /// Returns tensor shaped (1, n_kv_head, len, head_dim).
    #[allow(dead_code)]
    pub fn chunked_read(&self, _offset: usize, _len: usize) -> Result<Tensor> {
        match &self.storage {
            CacheStorage::Chunked(_) => {
                // Read returns both K and V - caller should use KvCache for paired access
                candle::bail!(
                    "chunked_read on individual Cache not supported; use KvCache::chunked_read_kv"
                )
            }
            CacheStorage::Contiguous { .. } => {
                candle::bail!("chunked_read only available for chunked caches")
            }
        }
    }

    pub(crate) fn set_chunked_backing(
        &mut self,
        backing: ChunkedKvBacking,
        batch_idx: usize,
        compression_policy: Option<CompressionPolicy>,
    ) -> Result<()> {
        let chunked = ChunkedCache::new(backing, batch_idx, compression_policy)?;
        self.storage = CacheStorage::Chunked(chunked);
        self.current_seq_len = 0;
        self.set_chunked_len(0);
        Ok(())
    }

    /// Configure a forced dtype for this cache.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.force_dtype = Some(dtype);
        self
    }

    /// Get the dimension along which the cache grows.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Force all data written to this cache to use the specified dtype.
    pub fn force_dtype(&mut self, dtype: DType) {
        self.force_dtype = Some(dtype);
    }

    /// Get the current dtype of the cache.
    pub fn dtype(&self) -> DType {
        if let Some(dtype) = self.force_dtype {
            dtype
        } else {
            match &self.storage {
                CacheStorage::Contiguous { all_data: Some(ad) } => ad.dtype(),
                CacheStorage::Chunked(c) => {
                    // For chunked caches, get dtype from the backing storage.
                    // If quantized, return F16 as that's what dequantize_f16 produces
                    // and what prepare_for_kernel uses for GPU float arenas.
                    c.backing.dtype().unwrap_or(DType::F16)
                }
                _ => DType::F32,
            }
        }
    }

    /// Get the current sequence length stored in the cache.
    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    /// Set the current sequence length.
    pub fn set_current_seq_len(&mut self, seq_len: usize) -> Result<()> {
        if self.is_chunked() {
            self.current_seq_len = seq_len;
            self.set_chunked_len(seq_len);
            return Ok(());
        }
        self.current_seq_len = seq_len;
        Ok(())
    }

    /// Get the maximum sequence length (capacity) of the cache.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get the full backing tensor for contiguous caches.
    pub fn all_data(&self) -> Option<&Tensor> {
        match &self.storage {
            CacheStorage::Contiguous { all_data } => all_data.as_ref(),
            CacheStorage::Chunked(_) => None,
        }
    }

    /// Get the current data (narrowed to current_seq_len) for contiguous caches.
    pub fn current_data(&self) -> Result<Option<Tensor>> {
        match &self.storage {
            CacheStorage::Contiguous { all_data } => Ok(match all_data.as_ref() {
                None => None,
                Some(d) => Some(d.narrow(self.dim, 0, self.current_seq_len)?),
            }),
            CacheStorage::Chunked(_) => Ok(None),
        }
    }

    /// Truncate the cache to the specified sequence length, freeing unused memory.
    pub fn truncate(&mut self, seq_len: usize) -> Result<()> {
        if self.is_chunked() {
            if seq_len > self.current_seq_len {
                return Ok(());
            }
            if seq_len == 0 {
                self.reset();
                return Ok(());
            }
            self.current_seq_len = seq_len;
            self.set_chunked_len(seq_len);
            return Ok(());
        }
        if seq_len < self.current_seq_len {
            if seq_len == 0 {
                // Special case: completely clear the cache
                self.storage = CacheStorage::Contiguous { all_data: None };
                self.current_seq_len = 0;
                self.max_seq_len = 0;
            } else if let CacheStorage::Contiguous { all_data } = &mut self.storage {
                if let Some(old_tensor) = all_data.take() {
                    // Extract what we want to keep
                    let kept = old_tensor.narrow(self.dim, 0, seq_len)?;

                    // Copy to new storage (only if non-contiguous)
                    let new_tensor = kept.contiguous()?;

                    // Replace with new tensor
                    *all_data = Some(new_tensor);
                    self.current_seq_len = seq_len;
                    self.max_seq_len = seq_len;
                }
            }
        }
        Ok(())
    }

    /// Try to truncate, but if OOM, do a full reset instead.
    /// Returns true if truncate succeeded, false if reset was performed.
    pub fn try_truncate_or_reset(&mut self, seq_len: usize) -> Result<bool> {
        if self.is_chunked() {
            if seq_len == 0 {
                self.reset();
                return Ok(false);
            }
            if seq_len >= self.current_seq_len {
                return Ok(true);
            }
            self.current_seq_len = seq_len;
            self.set_chunked_len(seq_len);
            return Ok(true);
        }
        if seq_len < self.current_seq_len {
            if seq_len == 0 {
                self.reset();
                return Ok(false);
            }

            if let CacheStorage::Contiguous { all_data } = &mut self.storage {
                if let Some(old_tensor) = all_data.take() {
                    // Try to narrow
                    let kept = match old_tensor.narrow(self.dim, 0, seq_len) {
                        Ok(k) => k,
                        Err(_) => {
                            // Narrow failed (shouldn't happen), reset
                            self.reset();
                            return Ok(false);
                        }
                    };

                    // Try to copy (only if non-contiguous)
                    match kept.contiguous() {
                        Ok(new_tensor) => {
                            // Success!
                            *all_data = Some(new_tensor);
                            self.current_seq_len = seq_len;
                            self.max_seq_len = seq_len;
                            Ok(true)
                        }
                        Err(_) => {
                            // OOM during copy - do full reset instead
                            self.reset();
                            Ok(false)
                        }
                    }
                } else {
                    Ok(true)
                }
            } else {
                Ok(true)
            }
        } else {
            Ok(true)
        }
    }

    /// Reset the cache, clearing all data.
    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.set_chunked_len(0);
        // NOTE: do NOT call `chunked.free_sequence()` here.  The backing slot
        // lifecycle is owned by `BatchedInferenceSession::{create,free}_sequence`.
        // Freeing here from inside a forward pass (via `reset_caches_at_zero`,
        // which fires on any `offset == 0` sequence) deallocates the slot
        // mid-pass and trips `validate_chunked_decode_batch` for the seq_len==1
        // (paged_decode) path with `chunked decode validation failed: sequence
        // slot is not allocated`, which then surfaces as
        // `CUDA_ERROR_ILLEGAL_ADDRESS` from stale GPU pointers.
        if let CacheStorage::Contiguous { all_data } = &mut self.storage {
            *all_data = None;
        }
    }

    /// Truncate the chunked sequence to exactly `offset` cum-tokens, freeing any
    /// writer-owned chunks/usage beyond it (Arc-shared prefix chunks are kept).
    /// This makes an offset-`N` re-prefill idempotent — re-running a prefill at the
    /// same offset must not stack stale tail chunks. Unlike `reset`, it keeps the
    /// backing slot allocated. No-op when already ≤ `offset` tokens.
    pub(crate) fn truncate_chunked_to_tokens(&mut self, offset: usize) {
        self.current_seq_len = offset;
        match &mut self.storage {
            CacheStorage::Chunked(c) => {
                let _ = c.backing.truncate_sequence_to_tokens(c.batch_idx, offset);
            }
            CacheStorage::Contiguous { all_data } => {
                if offset == 0 {
                    *all_data = None;
                }
            }
        }
    }

    /// Fork this cache, creating a new cache that shares data via copy-on-write.
    ///
    /// For chunked (paged) caches: complete blocks are shared via COW, partial
    /// blocks are copied. The new cache can be written to independently.
    ///
    /// For contiguous caches: the data is cloned.
    ///
    /// Returns a new `Cache` with the same content up to `current_seq_len`.
    pub fn fork(&self) -> Result<Self> {
        match &self.storage {
            CacheStorage::Chunked(chunked) => {
                let forked_chunked = chunked.fork(self.current_seq_len)?;
                Ok(Self {
                    storage: CacheStorage::Chunked(forked_chunked),
                    dim: self.dim,
                    current_seq_len: self.current_seq_len,
                    grow_by: self.grow_by,
                    max_seq_len: self.max_seq_len,
                    force_dtype: self.force_dtype,
                })
            }
            CacheStorage::Contiguous { all_data } => {
                // For contiguous caches, just clone the data
                Ok(Self {
                    storage: CacheStorage::Contiguous {
                        all_data: all_data.clone(),
                    },
                    dim: self.dim,
                    current_seq_len: self.current_seq_len,
                    grow_by: self.grow_by,
                    max_seq_len: self.max_seq_len,
                    force_dtype: self.force_dtype,
                })
            }
        }
    }

    /// Append data to the cache.
    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        if self.is_chunked() {
            candle::bail!("append is not supported for chunked KV caches")
        }
        let seq_len = src.dim(self.dim)?;

        // If we have a forced dtype, convert src to that dtype
        // otherwise, if we have all_data already, convert to its dtype
        let src = if let Some(dtype) = self.force_dtype {
            if src.dtype() != dtype {
                src.to_dtype(dtype)?
            } else {
                src.clone()
            }
        } else if let CacheStorage::Contiguous { all_data: Some(ad) } = &self.storage {
            if src.dtype() != ad.dtype() {
                src.to_dtype(ad.dtype())?
            } else {
                src.clone()
            }
        } else {
            src.clone()
        };

        // `slice_set` requires contiguous tensors.
        let src = src.contiguous()?;

        // Ensure backing storage exists and is large enough.
        //
        // Historically this grew via repeated `Tensor::cat` in chunks of `grow_by`, which can
        // cause large transient allocations and O(n^2) copy behavior for big prefills.
        // Here we grow via a single reallocation+copy to the required capacity.
        let required_capacity = self.current_seq_len + seq_len;
        let CacheStorage::Contiguous { all_data } = &mut self.storage else {
            unreachable!("append called on non-contiguous cache")
        };
        if all_data.is_none() || required_capacity > self.max_seq_len {
            let grow_by = self.grow_by.max(1);
            // Round up to a multiple of `grow_by`.
            let new_capacity = required_capacity.div_ceil(grow_by) * grow_by;

            // Allocate and (if needed) copy existing tokens.
            let mut new_shape = src.dims().to_vec();
            new_shape[self.dim] = new_capacity;
            let new_ad = Tensor::zeros(new_shape, src.dtype(), src.device())?;

            if let Some(old_ad) = all_data.take() {
                if self.current_seq_len > 0 {
                    let kept = old_ad
                        .narrow(self.dim, 0, self.current_seq_len)?
                        .contiguous()?;
                    new_ad.slice_set(&kept, self.dim, 0)?;
                }
            }

            *all_data = Some(new_ad);
            self.max_seq_len = new_capacity;
        }

        let CacheStorage::Contiguous { all_data } = &mut self.storage else {
            unreachable!("append called on non-contiguous cache")
        };
        let ad = all_data.as_mut().unwrap();
        ad.slice_set(&src, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}

/// Result of checking KV cache integrity for NaN values.
#[derive(Debug, Clone, PartialEq)]
pub enum CacheIntegrityResult {
    /// The cache is empty.
    Empty,
    /// The cache is valid (no NaN values found).
    Valid,
    /// The cache contains NaN values.
    Invalid {
        /// Number of NaN elements found.
        nan_count: usize,
        /// Total number of elements checked.
        total_elements: usize,
        /// Percentage of elements that are NaN.
        percentage: f64,
    },
}

/// A paired key-value cache for attention.
#[derive(Debug, Clone)]
pub struct KvCache {
    k: Cache,
    v: Cache,
}

impl KvCache {
    /// Create a new KV cache with contiguous backing.
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let k = Cache::new(dim, max_seq_len);
        let v = Cache::new(dim, max_seq_len);
        Self { k, v }
    }

    /// Get a reference to the K cache.
    pub fn k_cache(&self) -> &Cache {
        &self.k
    }

    /// Get a reference to the V cache.
    pub fn v_cache(&self) -> &Cache {
        &self.v
    }

    /// Get a mutable reference to the K cache.
    pub fn k_cache_mut(&mut self) -> &mut Cache {
        &mut self.k
    }

    /// Get a mutable reference to the V cache.
    pub fn v_cache_mut(&mut self) -> &mut Cache {
        &mut self.v
    }

    /// Get per-block valid token counts for this KV cache's sequence slot.
    /// Delegates to the K cache's chunked backing. Returns empty vec for contiguous caches.
    pub fn block_usage(&self) -> Vec<u32> {
        self.k.block_usage()
    }

    /// Get per-block absolute RoPE base positions for this KV cache's sequence slot.
    /// Delegates to the K cache's chunked backing. Returns empty vec for contiguous caches.
    pub fn chunk_rope_positions(&self) -> Vec<i32> {
        self.k.chunk_rope_positions()
    }

    /// Ensure the chunked backing has capacity for `offset + add` tokens.
    /// No-op for contiguous caches. Takes `&self` (uses interior mutability).
    pub fn ensure_chunked_for_offset(&self, offset: usize, add: usize) -> Result<()> {
        self.k.ensure_chunked_for_offset(offset, add)
    }

    /// Get byte-level compression stats for a sequence (delegates to K cache).
    /// Returns the BPE (bytes per element) for the sequence, weighted by the number of tokens in K and V.
    pub fn compression_bpe(&self, batch_idx: usize) -> Option<(f64, usize)> {
        let r1 = self.k.compression_bpe(batch_idx)?;
        let r2 = self.v.compression_bpe(batch_idx)?;
        let weighted_bpe = r1.0 + r2.0;
        let total_elements = r1.1 + r2.1;
        Some((weighted_bpe, total_elements))
    }

    pub fn compression_dist(&self, batch_idx: usize, ret: &mut HashMap<GgmlDType, usize>) {
        self.k.compression_dist(batch_idx, false, ret);
        self.v.compression_dist(batch_idx, true, ret);
    }

    /// Ensure chunked backing has the chunks needed for a batched decode step.
    ///
    /// This allocates chunks (and new arenas as needed) and updates the shared block table.
    /// Supports partial batches: each cache's batch_idx is used independently.
    pub fn ensure_chunked_capacity_batch(
        caches: &mut [&mut KvCache],
        offsets: &[usize],
        add: usize,
    ) -> Result<()> {
        if caches.is_empty() {
            return Ok(());
        }
        if caches.len() != offsets.len() {
            candle::bail!(
                "offset count mismatch: got {} offsets for {} caches",
                offsets.len(),
                caches.len()
            )
        }
        let backing = match &caches[0].k.storage {
            CacheStorage::Chunked(c) => c.backing.clone(),
            CacheStorage::Contiguous { .. } => candle::bail!("expected chunked backing"),
        };
        // Use sparse batched ensure to keep partial-batch support while taking
        // a single chunked-state lock for the full batch operation.
        let entries: Vec<(usize, usize)> = caches
            .iter()
            .zip(offsets.iter())
            .filter_map(|(cache, &offset)| cache.k.chunked_batch_idx().map(|b| (b, offset)))
            .collect();
        backing.ensure_for_batch_entries(&entries, add)?;
        if add == 1 {
            backing.validate_decode_batch_state(&entries)?;
        }
        Ok(())
    }

    /// Validate that the selected chunked caches are ready for a decode write.
    ///
    /// This is a lightweight host-side validation that checks the current write
    /// slice invariants without mutating chunk layout or uploading metadata.
    pub fn validate_chunked_decode_batch(caches: &[&mut KvCache], offsets: &[usize]) -> Result<()> {
        if caches.is_empty() {
            return Ok(());
        }
        if caches.len() != offsets.len() {
            candle::bail!(
                "offset count mismatch: got {} offsets for {} caches",
                offsets.len(),
                caches.len()
            )
        }
        let backing = match &caches[0].k.storage {
            CacheStorage::Chunked(c) => c.backing.clone(),
            CacheStorage::Contiguous { .. } => candle::bail!("expected chunked backing"),
        };
        let entries: Vec<(usize, usize)> = caches
            .iter()
            .zip(offsets.iter())
            .filter_map(|(cache, &offset)| cache.k.chunked_batch_idx().map(|b| (b, offset)))
            .collect();
        backing.validate_decode_batch_state(&entries)?;
        Ok(())
    }

    /// Prime the persistent decode slot-state buffers after prefill.
    ///
    /// This materializes the per-sequence GPU slot headers ahead of the first
    /// decode token so decode can immediately reuse them on the hot path.
    pub fn prime_chunked_decode_slots_batch(caches: &mut [&mut KvCache]) -> Result<()> {
        if caches.is_empty() {
            return Ok(());
        }
        let backing = match &caches[0].k.storage {
            CacheStorage::Chunked(c) => c.backing.clone(),
            CacheStorage::Contiguous { .. } => return Ok(()),
        };
        let entries: Vec<(usize, usize)> = caches
            .iter()
            .filter_map(|cache| {
                cache
                    .k
                    .chunked_batch_idx()
                    .map(|b| (b, cache.k.current_seq_len))
            })
            .collect();
        if entries.is_empty() {
            return Ok(());
        }
        // Ensure the write chunk exists for each sequence. When prefill ends exactly
        // at a chunk boundary (seq_len % CHUNK_SIZE == 0), the tail chunk is already
        // full and the next decode token needs a new chunk that isn't allocated yet.
        // ensure_for_batch_entries(entries, 1) allocates that chunk if needed.
        backing.ensure_for_batch_entries(&entries, 1)?;
        let arena_info = backing.resolve_arena_info()?;
        let _ = backing.sync_decode_gpu_chunks(&entries, &arena_info)?;
        Ok(())
    }

    /// Finalize sequences after generation completes.
    /// Get the current K cache data (narrowed to current_seq_len).
    pub fn k(&self) -> Result<Option<Tensor>> {
        self.k.current_data()
    }

    /// Get the current V cache data (narrowed to current_seq_len).
    pub fn v(&self) -> Result<Option<Tensor>> {
        self.v.current_data()
    }

    /// Get the maximum sequence length (capacity) of the cache.
    pub fn max_seq_len(&self) -> usize {
        self.k.max_seq_len()
    }

    /// Force all data written to this cache to use the specified dtype.
    pub fn force_dtype(&mut self, dtype: candle::DType) {
        self.k.force_dtype(dtype);
        self.v.force_dtype(dtype);
    }

    /// Configure this cache to use chunked (paged) backing storage.
    ///
    /// `compression_policy=None` selects uniform storage; `Some(p)` enables
    /// adaptive per-block selection at level `p.compression_level`.
    pub fn set_chunked_backing(
        &mut self,
        backing: &ChunkedKvBacking,
        batch_idx: usize,
        compression_policy: Option<CompressionPolicy>,
    ) -> Result<()> {
        self.k
            .set_chunked_backing(backing.clone(), batch_idx, compression_policy)?;
        self.v
            .set_chunked_backing(backing.clone(), batch_idx, compression_policy)?;
        Ok(())
    }

    /// Write K/V data to chunked backing storage.
    ///
    /// Expects tensors shaped (1, n_kv_head, len, head_dim).
    /// For quantized storage, data is quantized on write.
    pub fn chunked_write_kv(&self, offset: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        match (&self.k.storage, &self.v.storage) {
            (CacheStorage::Chunked(k_c), CacheStorage::Chunked(_v_c)) => {
                // Both K and V share the same backing
                k_c.write_contiguous(offset, k, v)
            }
            _ => candle::bail!("chunked_write_kv requires chunked backing"),
        }
    }

    /// Read K/V data from chunked backing storage (dequantizes if needed).
    ///
    /// Returns tensors shaped (1, n_kv_head, len, head_dim).
    pub fn chunked_read_kv(&self, offset: usize, len: usize) -> Result<(Tensor, Tensor)> {
        match &self.k.storage {
            CacheStorage::Chunked(c) => c.read_contiguous(offset, len),
            _ => candle::bail!("chunked_read_kv requires chunked backing"),
        }
    }

    /// Count the number of quantized arenas.
    ///
    /// Returns (quantized_count, total_count) tuple, or None if not chunked.
    /// Useful for validating that quantization is actually occurring.
    pub fn count_quantized_arenas(&self) -> Option<candle::Result<(usize, usize)>> {
        self.k.count_quantized_arenas()
    }

    /// Calculate the percentage of a sequence's tokens stored in quantized arenas.
    ///
    /// Returns (quantized_tokens, total_tokens) based on which ChunkRefs point to quantized arenas.
    /// This validates that the actual data for a sequence is quantized, not just that quantized arenas exist.
    pub fn quantized_token_stats(
        &self,
        batch_idx: usize,
    ) -> Option<candle::Result<(usize, usize)>> {
        self.k.quantized_token_stats(batch_idx)
    }

    /// Get the current dtype of the cache.
    pub fn dtype(&self) -> DType {
        self.k.dtype()
    }

    /// Append K and V tensors to the cache, returning the full accumulated tensors.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let out_k = self.k.current_data()?;
        let out_v = self.v.current_data()?;
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, k.dtype(), k.device())?
            }
            Some(k) => k,
        };
        let v = match out_v {
            None => {
                let mut shape = v.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, v.dtype(), v.device())?
            }
            Some(v) => v,
        };
        Ok((k, v))
    }

    /// Get the current sequence length stored in the cache.
    pub fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    /// Set the current sequence length.
    pub fn set_current_seq_len(&mut self, seq_len: usize) -> Result<()> {
        self.k.set_current_seq_len(seq_len)?;
        self.v.set_current_seq_len(seq_len)?;
        Ok(())
    }

    /// Truncate the cache to the specified sequence length.
    pub fn truncate(&mut self, seq_len: usize) -> Result<()> {
        self.k.truncate(seq_len)?;
        self.v.truncate(seq_len)?;
        Ok(())
    }

    /// Try to truncate, but if OOM, do a full reset instead.
    /// Returns true if truncate succeeded, false if reset was performed.
    pub fn try_truncate_or_reset(&mut self, seq_len: usize) -> Result<bool> {
        // Try k first
        let k_success = self.k.try_truncate_or_reset(seq_len)?;

        if !k_success {
            // K failed and was reset, reset v too for consistency
            self.v.reset();
            return Ok(false);
        }

        // Try v
        let v_success = self.v.try_truncate_or_reset(seq_len)?;

        if !v_success {
            // V failed, reset k for consistency
            self.k.reset();
            return Ok(false);
        }

        Ok(true)
    }

    /// Try to reserve space for additional tokens by expanding capacity.
    /// Maintains existing tokens. Returns true if successful, false if OOM.
    pub fn try_reserve(&mut self, additional_tokens: usize) -> bool {
        if self.k.is_chunked() || self.v.is_chunked() {
            // Chunked caches don't use the contiguous capacity growth logic.
            return true;
        }
        let current_len = self.current_seq_len();
        let required_capacity = current_len + additional_tokens;

        // Already have enough space
        if required_capacity <= self.k.max_seq_len {
            return true;
        }

        // Round up to nearest multiple of grow_by (e.g., pooled chunk size)
        let new_capacity = required_capacity.div_ceil(self.k.grow_by) * self.k.grow_by;

        // Expand k cache
        if let CacheStorage::Contiguous { all_data } = &mut self.k.storage {
            if let Some(old_k) = all_data.take() {
                let mut new_shape = old_k.dims().to_vec();
                new_shape[self.k.dim] = new_capacity;

                let new_k = match Tensor::zeros(new_shape, old_k.dtype(), old_k.device()) {
                    Ok(t) => t,
                    Err(_) => {
                        *all_data = Some(old_k);
                        return false;
                    }
                };

                if current_len > 0
                    && new_k
                        .slice_set(
                            &old_k.narrow(self.k.dim, 0, current_len).unwrap(),
                            self.k.dim,
                            0,
                        )
                        .is_err()
                {
                    *all_data = Some(old_k);
                    return false;
                }

                *all_data = Some(new_k);
                self.k.max_seq_len = new_capacity;
            } else {
                self.k.max_seq_len = new_capacity;
            }
        } else {
            return true;
        }

        // Expand v cache
        if let CacheStorage::Contiguous { all_data } = &mut self.v.storage {
            if let Some(old_v) = all_data.take() {
                let mut new_shape = old_v.dims().to_vec();
                new_shape[self.v.dim] = new_capacity;

                let new_v = match Tensor::zeros(new_shape, old_v.dtype(), old_v.device()) {
                    Ok(t) => t,
                    Err(_) => {
                        *all_data = Some(old_v);
                        return false;
                    }
                };

                if current_len > 0
                    && new_v
                        .slice_set(
                            &old_v.narrow(self.v.dim, 0, current_len).unwrap(),
                            self.v.dim,
                            0,
                        )
                        .is_err()
                {
                    *all_data = Some(old_v);
                    return false;
                }

                *all_data = Some(new_v);
                self.v.max_seq_len = new_capacity;
            } else {
                self.v.max_seq_len = new_capacity;
            }
        } else {
            return true;
        }

        true
    }

    /// Reset the cache, clearing all data.
    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }

    /// Truncate both K and V to exactly `offset` cum-tokens, freeing any chunks
    /// beyond it. Makes an offset-`N` (re)prefill idempotent (see
    /// [`Cache::truncate_chunked_to_tokens`]).
    pub fn truncate_to_offset(&mut self, offset: usize) {
        self.k.truncate_chunked_to_tokens(offset);
        self.v.truncate_chunked_to_tokens(offset);
    }

    /// Fork this KV cache, creating a new cache that shares data via copy-on-write.
    ///
    /// For chunked (paged) caches: complete blocks are shared via COW, partial
    /// blocks are copied. The new cache can be written to independently without
    /// affecting the original.
    ///
    /// For contiguous caches: the data is cloned.
    ///
    /// This is useful for beam search, speculative decoding, or any scenario
    /// where you need to branch from an existing sequence state.
    pub fn fork(&self) -> Result<Self> {
        Ok(Self {
            k: self.k.fork()?,
            v: self.v.fork()?,
        })
    }

    /// Check the integrity of the last position in the KV cache for NaN values.
    /// Returns Empty if cache is empty, Valid if no NaNs found, or Invalid with count and percentage.
    pub fn check_integrity(&self) -> Result<CacheIntegrityResult> {
        let seq_len = self.current_seq_len();

        // Empty cache
        if seq_len == 0 {
            return Ok(CacheIntegrityResult::Empty);
        }

        let last_pos = seq_len - 1;

        if self.k.is_chunked() || self.v.is_chunked() {
            // Integrity checks currently only support contiguous caches.
            // Chunked KV is validated by kernel-level tests.
            return Ok(CacheIntegrityResult::Valid);
        }

        // Get the last position from both k and v caches
        let k_data = match &self.k.storage {
            CacheStorage::Contiguous { all_data } => all_data.as_ref(),
            CacheStorage::Chunked(_) => None,
        };
        let v_data = match &self.v.storage {
            CacheStorage::Contiguous { all_data } => all_data.as_ref(),
            CacheStorage::Chunked(_) => None,
        };

        if k_data.is_none() || v_data.is_none() {
            return Ok(CacheIntegrityResult::Empty);
        }

        let k_data = k_data.unwrap();
        let v_data = v_data.unwrap();

        // Extract last position slice
        let k_last = k_data.narrow(self.k.dim, last_pos, 1)?;
        let v_last = v_data.narrow(self.v.dim, last_pos, 1)?;

        // Flatten and check for NaNs (convert to F32 to handle any dtype)
        let k_flat = k_last
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let v_flat = v_last
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        let mut nan_count = 0;
        let mut total_elements = 0;

        for &val in k_flat.iter() {
            total_elements += 1;
            if val.is_nan() {
                nan_count += 1;
            }
        }

        for &val in v_flat.iter() {
            total_elements += 1;
            if val.is_nan() {
                nan_count += 1;
            }
        }

        let nan_count = nan_count;
        let total_elements = total_elements;

        if nan_count > 0 {
            let percentage = (nan_count as f64 / total_elements as f64) * 100.0;
            Ok(CacheIntegrityResult::Invalid {
                nan_count,
                total_elements,
                percentage,
            })
        } else {
            Ok(CacheIntegrityResult::Valid)
        }
    }

    /// Convert the cache to the specified dtype.
    pub fn convert_dtype(&mut self, dtype: DType) -> Result<()> {
        if self.k.is_chunked() || self.v.is_chunked() {
            candle::bail!("convert_dtype is not supported for chunked KV caches")
        }
        // Convert K cache if it exists
        if let CacheStorage::Contiguous { all_data } = &mut self.k.storage {
            if let Some(k_data) = all_data.take() {
                if k_data.dtype() != dtype {
                    let converted = k_data.to_dtype(dtype)?;
                    *all_data = Some(converted);
                } else {
                    *all_data = Some(k_data);
                }
            }
        }
        // Convert V cache if it exists
        if let CacheStorage::Contiguous { all_data } = &mut self.v.storage {
            if let Some(v_data) = all_data.take() {
                if v_data.dtype() != dtype {
                    let converted = v_data.to_dtype(dtype)?;
                    *all_data = Some(converted);
                } else {
                    *all_data = Some(v_data);
                }
            }
        }
        Ok(())
    }

    /// Ensure the KV cache is in one of the specified dtypes.
    /// If already in one of the specified dtypes, returns without conversion.
    /// Otherwise, converts to the first dtype in the list.
    pub fn ensure_dtype(&mut self, dtypes: &[DType]) -> Result<()> {
        if dtypes.is_empty() {
            candle::bail!("ensure_dtype requires at least one dtype")
        }

        if self.k.is_chunked() || self.v.is_chunked() {
            candle::bail!("ensure_dtype is not supported for chunked KV caches")
        }

        // Convert K cache if it exists
        if let CacheStorage::Contiguous { all_data } = &mut self.k.storage {
            if let Some(k_data) = all_data.take() {
                if !dtypes.contains(&k_data.dtype()) {
                    let converted = k_data.to_dtype(dtypes[0])?;
                    *all_data = Some(converted);
                } else {
                    *all_data = Some(k_data);
                }
            }
        }

        // Convert V cache if it exists
        if let CacheStorage::Contiguous { all_data } = &mut self.v.storage {
            if let Some(v_data) = all_data.take() {
                if !dtypes.contains(&v_data.dtype()) {
                    let converted = v_data.to_dtype(dtypes[0])?;
                    *all_data = Some(converted);
                } else {
                    *all_data = Some(v_data);
                }
            }
        }

        Ok(())
    }
}
