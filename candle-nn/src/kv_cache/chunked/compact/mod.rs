//! The compaction pass: plan, copy, translate, publish.
//!
//! Packs every GPU KV pool toward the lowest physical addresses it can reach, so
//! the arena frontier falls and the weight side gets its ground back. A moved
//! chunk's gid is its location, so every holder of it, across this backing, the
//! substrate and the scheduler's caches, is rewritten in one window, or the pass
//! refuses rather than guesses. The order is fixed: census, copy, barrier,
//! rewrite, decode-buffer invalidation, and a device-wide sync before any arena
//! is released.
//!
//! - `plan`: the budget, the pool ranking and each pool's plan cap.
//! - `pass`: the pass itself, from census to publish.
//! - `copy`: the single-launch copy of moved slots, with its collision check.
//! - `verify`: the post-copy byte comparison and the own-holder rewrite.
//! - `provision`: fresh low arenas for the frontier's pools.
//! - `readonly`: source slots declared immutable for the pass.
//! - `report`: what a pass did.
//! - `tally`: cumulative outcomes since boot.
//! - `tests_compact`: planner, extent and collision tests.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use ahash::AHashSet;
use candle::Result;

use super::arena::{ArenaKey, ArenaKind};
use super::backing::ChunkedKvBacking;
use super::compact_map::{CompactionMap, Sweep};
use super::compact_plan::{by_source_rank, plan_pool, ArenaSlots, ChunkMove, CompactPlan};
use super::fresh_arenas::FreshArenas;
#[cfg(feature = "cuda")]
use super::gap::GapCycle;
#[cfg(feature = "cuda")]
use super::migrate_flight::retain_unheld;
use super::size_class::SizeClass;
#[cfg(feature = "cuda")]
use super::size_class::GID_STRIDE;
use super::types::StrandedRecords;
use crate::kv_cache::ArenaLocation;

mod copy;
mod pass;
mod plan;
mod provision;
mod readonly;
mod report;
mod tally;
#[cfg(test)]
mod tests_compact;
mod verify;

#[cfg(test)]
use copy::*;
pub use pass::*;
use plan::*;
#[cfg(all(feature = "cuda", feature = "tensor-assert"))]
use readonly::*;
pub use report::*;
pub use tally::*;
