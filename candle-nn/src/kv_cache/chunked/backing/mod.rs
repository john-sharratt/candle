//! Shared backing storage for the chunked KV cache.
//!
//! `ChunkedKvBacking` is the public handle, and each layer holds its own block
//! table over one shared `BackingInner`, so arenas are pooled across layers. The
//! same storage, record pool and allocator serve every layer. The record layout
//! types describe where each backing's records live and how large they are.
//!
//! - `constructors`: building a backing and its layers.
//! - `backing_inner`: the state shared by every layer of a group.
//! - `arena_census`: the census and ranking queries for compaction.
//! - `record_pool`: device KvHead records and their fill.
//! - `decode_slots`: cached decode slot buffers and their upload.
//! - `selection`: the select kernels' table and gid validation.
//! - `provenance`: sign(Q) sources and the sign-pack launch.
//! - `dump`: host decoding of stored bands for offline dumps.
//! - `compression_stats`: occupancy and bits-per-element diagnostics.
//! - `refusals`: markers for errors that mean no ground right now.
//! - `registry`: the process-wide list of backings.

use ahash::{HashMap, HashMapExt};
use candle::quantized::GgmlDType;
use std::borrow::Borrow;
use std::cmp;
use std::sync::{Arc, Mutex, RwLock, Weak};

use candle::quantized::pinned_staging::{Generation, PinnedStager};
// Core tensor types, needed on every build — the CPU one included. (These were
// gated on `cuda`, which is why a non-cuda build could not resolve `Device`,
// `DType`, or the crate's own `Result` anywhere in this file.)
use candle::{DType, Device, Result};

#[cfg(feature = "cuda")]
use super::compact_map::Replaced;
use super::compact_plan::ArenaSlots;
use super::fresh_arenas::FreshArenas;
use super::gap::LayerShape;
use super::head_gids::ChunkBands;
use super::size_class::{class_for_payload, payload_bytes_for_tag, SizeClass};
#[cfg(feature = "cuda")]
use super::slot_upload_batch::SlotUploadBatch;
use super::types::{ChunkWindow, DecodeGpuChunksSyncKind, ProvSource, QSign};
use super::{
    Arena, ArenaKey, ArenaStorage, ArenaStorageState, BlockTableState, ChunkMeta,
    CompressionPolicy, GpuArenaClassStats, LiveChunkRef, SealedChunk, StoragePolicy,
};
use crate::kv_cache::arena_table::{ArenaFormatTag, ArenaLocation, PerHeadEntry};
use std::collections::HashSet;
use std::time::Duration;

use super::gid_pool::ChunkGid;
// `N_PALETTE` is referenced by the intra-doc links throughout this file and by
// the CUDA table builders; without `cuda` only the doc links are left, and they
// still need it in scope to resolve.
#[cfg_attr(not(feature = "cuda"), allow(unused_imports))]
use crate::kv_cache::{HeadGids, KvFormat, QuantFormat, ResolvedArenaInfo, N_PALETTE};
use crate::{CHUNK_SIZE, GID_STRIDE};

/// Shared backing storage for a chunked (paged) KV cache.
///
/// Matches the addressing expected by the CUDA chunked ragged-attention kernels.
/// Supports both standard float storage and block-quantized (Q4_0, Q8_0) storage.
///
/// Multiple layers share the same `Arc<BackingInner>` (storage, pool)
/// but each layer has its own per-layer `state` (sequences, max_blocks).
#[derive(Debug, Clone)]
pub struct ChunkedKvBacking {
    pub(super) inner: Arc<BackingInner>,
    pub(crate) layer_idx: usize,
    /// Per-layer block table state (sequences, max_blocks).
    pub(crate) state: Arc<RwLock<BlockTableState>>,
}

/// Where a compaction's mints may land — see [`ChunkedKvBacking::try_alloc_record_slot`].
#[derive(Default)]
pub(super) struct RecordPlacement {
    /// Record arenas to fill first, lowest in the span first.
    pub order: Vec<usize>,
    /// `(arena, slot)` the pass has planned to read as record sources.
    pub avoid: HashSet<(usize, u32)>,
}

/// Where one backing's `KvHead` records live and how big they are — see
/// [`ChunkedKvBacking::record_layout`].
#[derive(Debug, Clone, Copy)]
pub(super) struct RecordLayout {
    /// The record arena pool for this geometry.
    pub key: ArenaKey,
    /// Serialized bytes of one chunk's whole `KvHead[n_kv_head]` record.
    pub record_bytes: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_palette: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeGpuChunkSyncStats {
    /// Slots serialised in full.
    pub rebuilds: u64,
    /// Slots whose live buffer served as it stood.
    pub reuses: u64,
    /// Slots whose stale writer region was re-serialised, then served.
    pub resyncs: u64,
    /// Entries those resyncs serialised, all slots together.
    pub resync_entries: u64,
    /// Slots extended over appended chunks — their tail serialised, then
    /// served.
    pub extends: u64,
    /// Entries those extends serialised, all slots together.
    pub extend_entries: u64,
    pub empty: u64,
    pub rebuild_time: Duration,
    pub reuse_time: Duration,
    pub resync_time: Duration,
    pub extend_time: Duration,
    /// Host time of the one batched upload that copies every slot the sync
    /// re-serialised.
    pub upload_time: Duration,
}

mod arena_census;
mod backing_inner;
mod compression_stats;
mod constructors;
mod decode_slots;
mod dump;
mod provenance;
mod record_pool;
mod refusals;
mod registry;
mod selection;

#[cfg(feature = "tensor-assert")]
pub use arena_census::*;
pub(crate) use backing_inner::*;
pub use refusals::*;
pub use registry::*;
