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
#[cfg(feature = "cuda")]
pub(crate) mod bump_arena;
mod chunk_ops;
mod compress;
mod compression_policy;
pub(super) mod cpu_selection;
pub mod fletcher_golden;
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
pub mod migrate_flight;
#[cfg(feature = "cuda")]
pub(crate) mod region_pool;
#[cfg(feature = "cuda")]
pub(crate) mod reservation;
pub mod sampled_selection;
mod sequence_ops;
mod size_class;
#[cfg(feature = "cuda")]
pub(crate) mod slot_state_arena;
mod types;

#[cfg(test)]
mod tests;

// Re-export public types
pub use backing::ChunkedKvBacking;
pub use backing::{global_arena_gpu_bytes, global_arena_memory_report, global_print_arena_table};
pub use backing::{is_device_oom, KV_DEVICE_OOM_MARKER};
pub use chunk_ops::BlockAllocSpec;
pub use chunk_ops::MIGRATION_STAGING_CAP_BYTES;
#[cfg(feature = "cuda")]
pub use compress::{
    convert_deferred_descs, dequantize_sealed_in_place, quantize_layers_deferred,
    quantize_sealed_in_place, quantize_sealed_in_place_deferred,
};
pub use compression_policy::{
    production_adaptive_candidates, CompressionPolicy, KvErrorThresholdFactors, LLAMA_KV_FACTORS,
    PRODUCTION_K_QREL_HIGH_THRESHOLDS, PRODUCTION_K_QREL_LOW_THRESHOLDS, PRODUCTION_LEVEL_TIER,
    PRODUCTION_V_QREL_HIGH_THRESHOLDS, PRODUCTION_V_QREL_LOW_THRESHOLDS, QWEN3_8B_KV_FACTORS,
    QWEN3_MOE_KV_FACTORS,
};
pub use gid_pool::{ChunkGid, ChunkGidPool, ClassOccupancy, GpuArenaClassStats};
pub use head_gids::HeadGids;
pub use meta_pool::MetaGid;
pub use migrate_flight::{migrate_flight, migrate_in_flight, MigrateFlight};
pub use size_class::{
    all_kv_formats, class_for_format, class_for_payload, elems_per_chunk, payload_bytes,
    payload_bytes_for_tag, SizeClass, GID_STRIDE, LADDER,
};
pub use types::{ChunkMeta, CHUNK_SIZE};
pub use types::{LiveChunkRef, SealedChunk, SealedSequence, WriterTail};

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

pub use alloc::class_promotion_count;
#[cfg(feature = "cuda")]
pub use bump_arena::{
    begin_wave, persistence_domain_stats, wave_alloc, wave_domain_stats, BumpRange,
    Generation as WaveGeneration,
};
#[cfg(feature = "cuda")]
pub use region_pool::{region_stats, RegionStats, REGION_BYTES};
#[cfg(feature = "cuda")]
pub use slot_state_arena::stats as slot_state_stats;
// Accurate KV VRAM budget query for the scheduler's budget-aware eviction.
#[cfg(feature = "cuda")]
pub use alloc::vram_budget_available;
