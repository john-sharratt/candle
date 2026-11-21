//! Chunk migration and format conversion between storage formats and between the
//! GPU and CPU locations.
//!
//! Chunks move between tiers and change format through this module. A migration
//! within a class is verbatim, because an arena is a run of untyped slots and
//! says nothing about its occupant. Format changes happen where the formats are
//! known, in the quantise and dequantise passes. Hot-to-warm migration is batched,
//! so the number of lock acquisitions and device copies does not grow with the
//! number of blocks.
//!
//! - `alloc_sealed`: allocation of sealed blocks and their chunk identifiers.
//! - `chunk_bytes`: reading, writing and relocating one chunk's payload.
//! - `migrate_layers`: warm copies from cold, and batched hot-to-warm migration.
//! - `tier_migration`: the staging cap and the staging groups.
//! - Tests: `tests_cpu_migration`, `tests_gpu_migration`, `tests_support`.

// Import from parent module (chunked/mod.rs)

use super::arena::ArenaStorageState;
#[cfg(feature = "cuda")]
use super::bump_arena;
#[cfg(feature = "cuda")]
use super::bump_arena::NOT_A_WAVE;
use super::gid_pool::ChunkGid;
use super::head_gids::{HeadGids, GIDS_PER_HEAD};
use super::meta_pool::ChunkRecordSrc;
#[cfg(feature = "cuda")]
use super::size_class::payload_bytes_for_tag;
use super::types::{SealedChunk, SealedSequence};
use super::{Arena, ArenaKey, ChunkedKvBacking};
use super::{CHUNK_SIZE, GID_STRIDE};
// Import from kv_cache module (grandparent)
#[cfg(feature = "cuda")]
use crate::kv_cache::arena_table::ArenaFormatTag;
use crate::kv_cache::arena_table::{ArenaLocation, N_PALETTE};
use crate::kv_cache::KvFormat;
#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::CudaStream;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::SELECT_FMT_F16;
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::PinnedBuf;
#[cfg(feature = "cuda")]
use candle::Device;
use candle::Result;
#[cfg(feature = "cuda")]
use std::sync::Arc;

// ── Tests ─────────────────────────────────────────────────────────────────────

mod alloc_sealed;
mod chunk_bytes;
mod migrate_layers;
mod tier_migration;
#[cfg(all(test, feature = "cuda"))]
use chunk_bytes::*;
#[cfg(test)]
mod tests_cpu_migration;
#[cfg(test)]
mod tests_gpu_migration;
#[cfg(test)]
mod tests_support;

pub use alloc_sealed::*;
pub use migrate_layers::*;
pub use tier_migration::*;
