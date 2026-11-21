//! Chunk slot allocation from per-format arena pools, with reference-counted gids.
//!
//! Each gid holds one reference to one slot, and an allocation takes the
//! lowest-placed arena with room, so the reservation fills from its low end.
//! Placement follows the region rank rather than the arena index, because a
//! recycled index says nothing about where its arena sits.
//!
//! - `arena_pool`: per-format arenas in placement order, with their capacity bitmap.
//! - `gid`: the typed reference that owns one slot.
//! - `pool`: the pool of slots by key, with index recycling.
//! - `refcount`: the lock-free refcount table of one arena.
//! - `stats`: occupancy counts by arena and size class.
//! - `tests_claim`: where a claim lands.
//! - `tests_gid`: gid lifetimes and concurrent allocation.
//! - `tests_lifecycle`: when storage may be released.
//! - `tests_reserve`: the reserve each class keeps.
//! - `tests_stats`: the per-class occupancy report.
//! - `tests_support`: shared pool fixtures.

use ahash::{AHashMap, AHashSet};
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::{
    collections::VecDeque,
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, RwLock,
    },
};
use strum::IntoEnumIterator;

use super::arena::{ArenaKey, ArenaKind, RECORD_STRIDES};
use super::size_class::SizeClass;
use crate::kv_cache::chunked::types::{GID_STRIDE, TARGET_ARENA_BYTES};
use crate::kv_cache::ArenaLocation;

/// The class the pool's own convenience allocator and its unit tests use.
///
/// Rung 5 is **640 B** (`Q4_KS`), which puts 26,214 slots in a 16 MiB region —
/// well above the 15,420 of `Q8_0`'s 1088 B rung. The point is to exercise the
/// free list and the `u16` recycle links near their busiest, not to mirror the
/// most common sealed format, so a *small* class is the stronger choice here.
///
/// (This said "2048 B … holds `Q8_0`". Both halves were wrong: rung 5 is 640 B,
/// and 2048 B is the `F16`/`BF16` rung — `Q8_0` sits at 1088.)
const TEST_CLASS: SizeClass = SizeClass::at(5);

mod arena_pool;
mod gid;
mod pool;
mod refcount;
mod stats;
#[cfg(test)]
mod tests_claim;
#[cfg(test)]
mod tests_gid;
#[cfg(test)]
mod tests_lifecycle;
#[cfg(test)]
mod tests_reserve;
#[cfg(test)]
mod tests_stats;
#[cfg(test)]
mod tests_support;

use arena_pool::*;
pub use gid::*;
pub use pool::*;
pub use refcount::*;
pub use stats::*;
