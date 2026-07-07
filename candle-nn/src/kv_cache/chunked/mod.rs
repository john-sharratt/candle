//! Chunked (paged) KV cache backing storage.
//!
//! This module provides `ChunkedKvBacking`, which implements paged KV cache storage
//! with support for:
//! - Arc-based prefix sharing (COW - copy-on-write)
//! - Per-sequence slot allocation
//! - Efficient memory reuse via free lists
//! - Cooperative arena compaction for memory pressure relief
//! - **Quantized storage (Q4_0, Q8_0)** for memory savings
//!
//! # Module Structure
//!
//! - `types` - Core types: ChunkHandle, ChunkRef, SlotState, ChunkedState
//! - `arena` - Arena storage: Arena, ArenaKey, ArenaStorage, StoragePolicy
//! - `backing` - Main implementation: BackingInner, ChunkedKvBacking, registry
//! - `alloc` - Allocation: ensure_max_blocks, create_arena, alloc_chunk, ensure_for_*
//! - `io` - I/O operations: read_contiguous, write_contiguous
//! - `chunk_ops` - Chunk operations: migrate_chunk, copy_chunk_data, prepare, reconcile
//! - `sequence_ops` - Sequence operations: alloc_sequence, free_sequence, share_prefix, fork_sequence

// Submodules
mod alloc;
mod arena;
mod backing;
mod chunk_ops;
mod compress;
mod compression_policy;
pub(super) mod cpu_selection;
mod gid_pool;
#[cfg(feature = "cuda")]
mod gpu_chunks;
#[cfg(not(feature = "cuda"))]
#[path = "gpu_chunks_dummy.rs"]
mod gpu_chunks;
mod head_gids;
mod io;
mod meta_pool;
pub mod migrate;
pub mod sampled_selection;
mod sequence_ops;
mod types;

#[cfg(test)]
mod tests;

// Re-export public types
pub use backing::ChunkedKvBacking;
pub use backing::{global_arena_gpu_bytes, global_arena_memory_report, global_print_arena_table};
pub use backing::{is_device_oom, KV_DEVICE_OOM_MARKER};
pub use chunk_ops::BlockAllocSpec;
#[cfg(feature = "cuda")]
pub use compress::{dequantize_sealed_in_place, quantize_sealed_in_place};
pub use compression_policy::{
    production_adaptive_candidates, CompressionPolicy, KvErrorThresholdFactors, LLAMA_KV_FACTORS,
    PRODUCTION_K_QREL_HIGH_THRESHOLDS, PRODUCTION_K_QREL_LOW_THRESHOLDS, PRODUCTION_LEVEL_TIER,
    PRODUCTION_V_QREL_HIGH_THRESHOLDS, PRODUCTION_V_QREL_LOW_THRESHOLDS, QWEN3_8B_KV_FACTORS,
    QWEN3_MOE_KV_FACTORS,
};
pub use gid_pool::{ChunkGid, ChunkGidPool};
pub use head_gids::HeadGids;
pub use meta_pool::MetaGid;
pub use types::{arena_chunks_for_format, arena_gid_stride, ChunkMeta, CHUNK_SIZE};
pub use types::{SealedChunk, SealedSequence, WriterTail};

// Re-export for use within submodules and tests
pub use arena::ArenaKey;
pub use arena::StoragePolicy;
pub(crate) use arena::{Arena, ArenaStorage, ArenaStorageState};
#[allow(unused_imports)]
pub(crate) use types::{BlockTableState, ChunkWindow, SequenceState};

// Import arena_table types for submodule use, and re-export
// `ArenaLocation` so callers can construct `SealedSequence` (whose
// `location` field is the coarse-grained tier tag).
pub use super::arena_table::ArenaLocation;

// Accurate KV VRAM budget query for the scheduler's budget-aware eviction.
#[cfg(feature = "cuda")]
pub use alloc::vram_budget_available;
