//! The chunked KV cache for one sequence slot, a wrapper over the backing that owns
//! the slot's arena storage.
//!
//! A slot is allocated in the backing before any offset is ensured. The cache reads
//! and writes its K and V through that slot, and the compression policy is decided
//! by its owner, where a missing policy means uniform storage. Forking shares complete
//! blocks copy-on-write, and the chunk references carry the RoPE shift that positions
//! them. A slot-state sync returns the serialised slice array the decode kernels walk,
//! and reports for each slot whether it was rebuilt, resynced or served as it stood,
//! so the cost of a sync is visible.

use super::*;

/// What a slot-state sync hands back per slot: the serialised slice array's
/// `(slices_ptr, n_slices, write_slice)`, the pins keeping the chunks it
/// references alive, and how each slot was brought up to date — rebuilt in
/// full, resynced in its writer region, or served as it stood — with the time
/// each kind took.
pub type SlotStateSync = (
    Vec<(u64, u32, u32)>,
    Vec<Arc<Vec<ChunkPin>>>,
    DecodeGpuChunkSyncStats,
);

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
    pub(crate) fn storage_policy(&self) -> super::super::StoragePolicy {
        self.backing.storage_policy()
    }

    /// Get the max_blocks for this cache.
    pub(crate) fn max_blocks(&self) -> usize {
        self.backing.max_blocks()
    }

    /// Execute a read operation on the arena storage.
    pub(crate) fn with_arenas<R>(
        &self,
        f: impl FnOnce(&ahash::AHashMap<usize, super::super::Arena>) -> R,
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
            compression_policy: self.compression_policy,
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
    ) -> Result<Vec<super::super::HeadGids>> {
        self.backing.get_chunk_refs(self.batch_idx, block_range)
    }

    /// Get chunk references with RoPE shift.
    pub(crate) fn get_chunk_refs_with_rope(
        &self,
        block_range: Option<std::ops::Range<usize>>,
    ) -> Result<Vec<super::super::HeadGids>> {
        self.backing
            .get_chunk_refs_with_rope(self.batch_idx, block_range)
    }

    /// Write contiguous K/V data to the chunked backing.
    /// Expects tensors shaped (1, n_kv_head, len, head_dim).
    pub(crate) fn write_contiguous(
        &self,
        offset: usize,
        k: &candle::LiveTensor<'_>,
        v: &candle::LiveTensor<'_>,
    ) -> Result<()> {
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
