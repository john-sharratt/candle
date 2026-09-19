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

// The arena layer's functions are transcriptions of CUDA kernel launches —
// pointers, extents, strides, formats, stream — and its queries return the raw
// multi-part tuples the kernels and the block tables are expressed in. Grouping
// either into structs would put a shape between this code and the layout it
// exists to describe, which is what makes it auditable against the `.cu` and the
// `KvHead` record. Same reasoning as `candle_core::quantized::cuda`.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

// Submodules
mod alloc;
mod arena;
mod backing;
/// Band addresses from the host block table, for captures that read KV raw.
#[cfg(feature = "tensor-assert")]
mod band_map;
#[cfg(feature = "cuda")]
pub(crate) mod bump_arena;
mod chunk_ops;
// Quantize-on-evict drives the palette-4 selection/convert CUDA kernels end to
// end — every one of its imports is already `cfg(cuda)`, and there is no CPU
// form of it. Gated whole rather than shot through with per-item cfgs.
#[cfg(feature = "cuda")]
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
#[cfg(all(test, feature = "cuda"))]
mod gpu_test_lock;
/// The growth direction's decision, split out so a trajectory can be run without
/// a device — see `docs/vram_partition_behavioural_tests.md`.
pub mod growth_policy;
#[cfg(feature = "cuda")]
pub mod guard;
#[cfg(not(feature = "cuda"))]
mod guest_stage_cpu;
mod head_gids;
mod io;
mod meta_pool;
pub mod migrate;
pub mod migrate_flight;
#[cfg(feature = "cuda")]
pub(crate) mod region_pool;
/// Owner-down chunk relocation: the transform that actually moves bytes.
#[cfg(feature = "cuda")]
pub mod relocate;
/// Which arenas a relocation pass should empty. Pure arithmetic over the arena
/// census, and outside the `cuda` gate for the reason the span geometry is: the
/// policy's defects are trajectory defects, provable with no device in reach.
pub mod relocate_plan;
#[cfg(feature = "cuda")]
pub(crate) mod reservation;
pub mod sampled_selection;
mod sequence_ops;
mod size_class;
#[cfg(feature = "cuda")]
pub(crate) mod slot_state_arena;
/// The KV side's extents, published to the between-waves overlap audit.
/// Gated with the audit itself — without `tensor-assert` there is nothing to
/// publish to.
#[cfg(feature = "tensor-assert")]
pub mod span_claims;
/// Checks a live decode row's offset against the layer it is serialised for.
#[cfg(feature = "tensor-assert")]
mod writer_len_audit;
#[cfg(feature = "tensor-assert")]
pub use band_map::{BandAddr, BlockBands, WriterIndices};
#[cfg(feature = "tensor-assert")]
pub use types::BlockTableMutation;
/// Where the tier may stand and what the KV side may reach. Pure arithmetic, and
/// outside the `cuda` gate so it can be exercised on any machine.
pub mod span_geometry;
mod types;
// Instrumentation for the bump arenas' high-water marks: its only caller is
// `bump_arena`, so it shares that module's gating.
#[cfg(feature = "cuda")]
mod wave_census;
pub mod wave_plan;
/// Unconditional: the phase spans are measurements, and everything that reads
/// them — the region carve, the row pricing — is integer arithmetic.
pub mod wave_spans;
/// Unconditional: the zone is pure arithmetic over addresses and slot indices,
/// so its invariants — the ones a mis-set boundary would violate — are provable
/// on a machine with no GPU.
pub mod weight_zone;

#[cfg(test)]
mod tests;

// Re-export public types
pub use backing::ChunkedKvBacking;
pub use backing::{
    global_arena_gpu_bytes, global_arena_map, global_arena_memory_report, global_print_arena_table,
};
pub use backing::{is_device_oom, is_tier_refusal, KV_DEVICE_OOM_MARKER, TIER_REFUSAL_MARKER};
pub use chunk_ops::BlockAllocSpec;
pub use chunk_ops::MIGRATION_STAGING_CAP_BYTES;
#[cfg(feature = "cuda")]
pub use compress::{
    convert_deferred_descs, dequantize_sealed_in_place, quantize_layers_deferred,
    quantize_sealed_in_place, quantize_sealed_in_place_deferred,
};
pub use compression_policy::{
    production_adaptive_candidates, CompressionPolicy, KvErrorThresholdFactors, LLAMA2_KV_FACTOR,
    LLAMA3_KV_FACTOR, LLAMA_KV_FACTORS, PRODUCTION_K_QREL_HIGH_THRESHOLDS,
    PRODUCTION_K_QREL_LOW_THRESHOLDS, PRODUCTION_LEVEL_TIER, PRODUCTION_V_QREL_HIGH_THRESHOLDS,
    PRODUCTION_V_QREL_LOW_THRESHOLDS, QWEN35_0_8B_KV_FACTORS, QWEN35_9B_KV_FACTORS,
    QWEN35_MOE_KV_FACTORS, QWEN36_MOE_KV_FACTORS, QWEN38_KV_FACTORS, QWEN3_8B_KV_FACTORS,
    QWEN3_MOE_KV_FACTORS, QWEN4EXP_KV_FACTORS, QWEN4EXP_Q2KO_KV_FACTORS,
};
pub use gid_pool::{ArenaOccupancy, ChunkGid, ChunkGidPool, ClassOccupancy, GpuArenaClassStats};
pub use head_gids::HeadGids;
pub use meta_pool::MetaGid;
pub use migrate_flight::{migrate_flight, migrate_in_flight, MigrateFlight};
#[cfg(feature = "cuda")]
pub use relocate::RelocationOutcome;
pub use relocate_plan::{plan_class as plan_relocation_class, RelocationPlan, MIN_RELOCATION_GAIN};
pub use size_class::{
    all_kv_formats, class_for_format, class_for_payload, elems_per_chunk, payload_bytes,
    payload_bytes_for_tag, SizeClass, GID_STRIDE, LADDER,
};
pub use types::{ChunkMeta, CHUNK_SIZE};
pub use types::{LiveChunkRef, SealedChunk, SealedSequence, WriterTail};
pub use weight_zone::{
    RetractPlan, WeightZone, WeightZoneStats, INITIAL_KV_RESERVE, MIN_ELASTIC_RESERVE,
};

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
    begin_forward, begin_guest, begin_wave, close_guest_arena, end_wave_transient,
    guest_domain_stats, guest_stage, open_guest_arena, persistence_domain_stats,
    plan_wave_transient, wave_domain_stats, wave_is_live, wave_max_planned, wave_max_slack,
    wave_reset_observations, wave_settle, wave_worst_slack, BumpRange, ForwardOpen,
    Generation as WaveGeneration, GUEST_ARENA, KV_ARENA_MID_WAVE,
};
#[cfg(feature = "cuda")]
pub use guard::{expect_kv_range, expect_kv_range_in};
#[cfg(not(feature = "cuda"))]
pub use guest_stage_cpu::guest_stage;
#[cfg(feature = "cuda")]
pub use region_pool::{
    claim_dense, claim_span_region, dense_bytes, empty_sweep_stats, ensure_reservation,
    freeze_dense, initial_weight_bytes, kv_spare_regions, least_tier_bytes, reclaim_empty_arenas,
    reclaim_load_headroom, region_stats, set_least_tier_bytes, set_weight_floor, span_end,
    span_layout, span_region_refusal, spare_tally, transient_headroom_bytes, weight_capacity_bytes,
    weight_floor_after, RegionStats, SpanClaims, SpanLayout, SpanRegion, REGION_BYTES,
};
#[cfg(feature = "cuda")]
pub use slot_state_arena::stats as slot_state_stats;
pub use wave_spans::{WAVE_ATTN_BYTES, WAVE_FFN_BYTES, WAVE_FORWARD_BYTES, WAVE_SPAN_BYTES};
// Accurate KV VRAM budget query for the scheduler's budget-aware eviction.
// Defined in both configurations — `None` when there is no CUDA device to
// budget — so the export is unconditional too.
pub use alloc::vram_budget_available;
