//! Internal allocation methods for a chunked KV backing: arena creation, chunk
//! claiming, and the writes that make a sequence's tail writable.
//!
//! The chunks of a backing are slots in arenas, slabs carved from the device
//! reservation or held in host memory. A claim takes a slot from an arena with
//! room; one that finds none records what it needs, and the arena is created in
//! the gap between forwards. Writes first make the sequence's tail writable, and
//! the layers of a sequence are kept at one shape.
//!
//! - `arena_window`: the gate that defers arena creation to the gap.
//! - `budget`: the exact count of KV bytes the reservation can still hold.
//! - `chunk_alloc`: chunk allocation from free lists and new arenas.
//! - `claim`: claiming a slot, and promoting a starved class to a wider one.
//! - `counters`: process-wide creation and promotion counters.
//! - `ensure_for_write`: making a sequence's tail writable before a write.
//! - `ensure_layout`: keeping every layer's chunk layout the same.
//! - `reserves`: the reserve each class keeps free for conversions.
//! - `tests_owe`: the owed-arena arithmetic.

use std::cmp;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

// `wave_provenance` lives at the candle-core root and is not CUDA-gated; only
// the `cuda_backend` re-export of it is. Reaching it through that re-export
// made this file unbuildable without the `cuda` feature.
#[cfg(feature = "cuda")]
use candle::wave_provenance::LeaseOrigin;
use candle::{DType, Device, Error, Result, Tensor};

use super::arena::{ArenaKey, ArenaKind};
use super::backing::ChunkedKvBacking;
#[cfg(feature = "cuda")]
use super::backing::{KV_DEVICE_OOM_MARKER, TIER_STANDS_OVER_FREE};
#[cfg(feature = "cuda")]
use super::bump_arena::{enter_arena_window, ArenaWindow, KV_ARENA_MID_WAVE};
use super::gid_pool::ChunkGid;
use super::head_gids::HeadGids;
#[cfg(feature = "cuda")]
use super::region_pool;
use super::size_class::{elems_per_chunk, SizeClass};
use super::types::{ChunkWindow, DecodeLayout, CHUNK_SIZE};
use super::write_placement::appended_tokens;
use super::{Arena, ArenaLocation};
use crate::kv_cache::arena_table::ArenaFormatTag;
use crate::kv_cache::chunked::backing::BackingInner;
use crate::kv_cache::chunked::ArenaStorageState;
use crate::kv_cache::{KvFormat, QuantFormat};

mod arena_window;
mod budget;
mod counters;
mod reserves;
#[cfg(all(test, feature = "cuda"))]
use arena_window::*;
mod chunk_alloc;
mod claim;
mod ensure_for_write;
mod ensure_layout;
#[cfg(test)]
mod tests_owe;

pub use budget::*;
pub use counters::*;
