//! Non-CUDA stub for [`GpuChunks`].
//!
//! Provides the same public surface as `gpu_chunks.rs` but all operations
//! are no-ops.  Selected at compile time when the `cuda` feature is absent.

use super::types::ChunkWindow;
use crate::kv_cache::arena_table::ResolvedArenaInfo;

/// Serialised byte-size of one `TokenSlice` entry (mirrors the real implementation).
#[allow(dead_code)]
pub(crate) fn token_slice_serialized_size(_n_kv_head: usize, _head_dim: usize) -> usize {
    0
}

/// No-op stub for serialize_chunk_window_with_len (CUDA-only in real impl).
#[allow(dead_code)]
pub(crate) fn serialize_chunk_window_with_len(
    _chunk: &ChunkWindow,
    _n_kv_head: usize,
    _head_dim: usize,
    _rope_base: u32,
    _len: u16,
    _arena_info: &[ResolvedArenaInfo],
    _dst: &mut [u8],
) {
}

/// No-op stand-in for the real CUDA-backed slot-state cache.
#[derive(Clone)]
pub(crate) struct GpuChunks;

impl std::fmt::Debug for GpuChunks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuChunks").finish()
    }
}

impl GpuChunks {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn as_mut(&mut self) -> GpuChunksGuard<'_> {
        GpuChunksGuard { _inner: self }
    }

    pub(crate) fn raw_device_ptr(&self) -> u64 {
        0
    }

    pub(crate) fn n_chunks(&self) -> usize {
        0
    }
}

pub(crate) struct GpuChunksGuard<'a> {
    _inner: &'a mut GpuChunks,
}

impl GpuChunksGuard<'_> {
    pub(crate) fn update_chunk(
        &mut self,
        _chunk_idx: usize,
        _chunk: &ChunkWindow,
        _n_kv_head: usize,
        _head_dim: usize,
        _rope_base: u32,
        _arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<()> {
        Ok(())
    }

    pub(crate) fn clear(&mut self) {}

    pub(crate) fn rebuild_decode(
        &mut self,
        _chunks: &[ChunkWindow],
        _n_kv_head: usize,
        _head_dim: usize,
        _arena_info: &[ResolvedArenaInfo],
        _write_len: u16,
    ) -> candle::Result<()> {
        Ok(())
    }
}
