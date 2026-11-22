//! The single-tensor and paired caches that let an attention layer use contiguous or
//! chunked KV storage behind one interface.
//!
//! A `Cache` is one side of the storage, K or V. It reports its chunked backing, slot
//! and format only when it has one, and contiguous storage answers those with empty
//! values. Chunked caches are reconciled to the backing's storage policy after each
//! kernel. Truncation is fallible and its failure reaches the caller, since layers
//! truncated to different lengths cannot be read by one decode position map. A fork
//! shares complete blocks copy-on-write, and a contiguous cache's fork clones its data.
//!
//! - `chunked_cache`: the chunked cache for one sequence slot.
//! - `kv_cache_pair`: the paired K and V cache an attention layer uses.

use super::chunked::{
    ChunkPin, ChunkedKvBacking, CompressionPolicy, DecodeGpuChunkSyncStats, CHUNK_SIZE, GID_STRIDE,
};
use ahash::HashMap;
use candle::quantized::pinned_staging::Generation;
use candle::quantized::GgmlDType;
use candle::{DType, Result, Tensor};
use std::sync::Arc;

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

    /// The arena slot this cache is bound to, if it is chunked.
    ///
    /// The companion to [`Self::chunked_backing`]: the pair `(backing, slot)`
    /// is what every arena call is keyed on — allocation, sealing, layout — so
    /// exposing one without the other forces callers to carry the slot
    /// alongside a cache that already knows it.
    pub fn chunked_slot(&self) -> Option<usize> {
        match &self.storage {
            CacheStorage::Chunked(c) => Some(c.batch_idx),
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

    /// Get the number of chunks per arena for chunked caches.
    pub fn chunked_arena_chunks(&self) -> Option<usize> {
        match &self.storage {
            CacheStorage::Chunked(_) => Some(GID_STRIDE),
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
                Some((bpe, n))
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

    /// [`Self::compression_bpe`] for one side — K when `is_value` is false.
    /// `None` for contiguous storage.
    pub fn compression_bpe_side(&self, batch_idx: usize, is_value: bool) -> Option<(f64, usize)> {
        match &self.storage {
            CacheStorage::Chunked(c) => c.backing.compression_bpe_side(batch_idx, is_value).ok(),
            CacheStorage::Contiguous { .. } => None,
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

    /// Get sealed chunk descriptors for this cache slot's live sequence.
    ///
    /// Returns `None` for contiguous caches or if the backing has no live sequence
    /// for this slot's batch index.
    pub fn chunked_live_chunks_as_sealed(&self) -> Option<Vec<super::SealedChunk>> {
        match &self.storage {
            CacheStorage::Chunked(c) => c.backing.live_chunks_as_sealed(c.batch_idx),
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
                let chunks = c.backing.live_chunks_as_sealed(c.batch_idx)?;
                Some(c.backing.dump_sealed_to_host(&chunks, device))
            }
            CacheStorage::Contiguous { .. } => None,
        }
    }

    /// Visit this slot's live chunks as zero-clone [`super::LiveChunkRef`]s
    /// (see `ChunkedKvBacking::visit_live_chunks`). `None` for contiguous
    /// storage or an unallocated slot.
    pub fn chunked_visit_live_chunks<R>(
        &self,
        f: impl FnOnce(&mut dyn Iterator<Item = super::LiveChunkRef<'_>>) -> R,
    ) -> Option<R> {
        match &self.storage {
            CacheStorage::Chunked(c) => c.backing.visit_live_chunks(c.batch_idx, f),
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
                    //
                    // F16 for a quantized backing is **not** an arbitrary
                    // fallback: the quantized formats are the *sealed* storage,
                    // while a live sequence's K sits in `R16` — raw F16 with
                    // Q-capture space — and its V in plain F16. So this reports
                    // the dtype of the bytes the forward actually reads.
                    // Answering BF16 here instead makes the declared activation
                    // dtype disagree with the live arena, which the engine
                    // refuses outright: "K and V arena formats require different
                    // compute dtypes: K=R16(BF16) V=F16(F16)".
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
    /// Cut this layer's chunked storage back to `offset` cum-tokens.
    ///
    /// **Fallible, and the failure must reach the caller.** This is called once
    /// per layer with a common offset (`admit_wave_kv`), so a failure that is
    /// swallowed here does not truncate one layer while the other forty-seven
    /// truncate — and the host-side `current_seq_len` is updated either way, so
    /// nothing downstream can tell. The layers then hold different token windows
    /// for the same sequence, which no single decode position map can describe;
    /// it surfaced as `chunked decode layout diverged across layers … First
    /// difference at chunk 67: (offset 0, usage 26) … (offset 0, usage 25)` on
    /// the first decode after a fork.
    ///
    /// The length is set only once the storage agrees with it, for the same
    /// reason: a length that describes a truncation that did not happen is worse
    /// than no truncation, because it is not detectable.
    pub(crate) fn truncate_chunked_to_tokens(&mut self, offset: usize) -> Result<()> {
        match &mut self.storage {
            CacheStorage::Chunked(c) => {
                c.backing.truncate_sequence_to_tokens(c.batch_idx, offset)?;
            }
            CacheStorage::Contiguous { all_data } => {
                if offset == 0 {
                    *all_data = None;
                }
            }
        }
        self.current_seq_len = offset;
        Ok(())
    }

    /// Take back a failed forward's writes to this layer: its chunked storage cut
    /// back to `offset` as [`Self::truncate_chunked_to_tokens`] cuts it, by
    /// [`ChunkedKvBacking::rollback_sequence_to_tokens`].
    pub(crate) fn rollback_chunked_to_tokens(&mut self, offset: usize) -> Result<()> {
        match &mut self.storage {
            CacheStorage::Chunked(c) => {
                c.backing.rollback_sequence_to_tokens(c.batch_idx, offset)?;
            }
            CacheStorage::Contiguous { all_data } => {
                if offset == 0 {
                    *all_data = None;
                }
            }
        }
        self.current_seq_len = offset;
        Ok(())
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

mod chunked_cache;
mod kv_cache_pair;

pub use chunked_cache::*;
pub use kv_cache_pair::*;
