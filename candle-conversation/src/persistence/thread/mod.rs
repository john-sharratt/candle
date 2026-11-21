//! The substrate persistence thread: the single owner of the redo-log write
//! path, one per workspace.
//!
//! The scheduler spawns the thread at engine start. It wakes on a periodic tick
//! or on a trigger the scheduler sends after each turn is sealed, and each wake
//! runs one pass over the substrate. The thread copies and writes but quantizes
//! nothing, since a turn's K/V is quantized on the scheduler thread and a turn
//! that is not yet finished is left alone. Shutdown drains the substrate to
//! durable storage before the thread is joined.
//!
//! - `drive`: the loop, its wake rules and the shutdown drain.
//! - `gauge`: the backlog gauges the scheduler throttles on.
//! - `handle`: the trigger and the readers.
//! - `pass`: the pass's phases.
//! - `warm_copy`: the hot-to-warm copy and the turn policy.
//! - `tests_thread`: tests of the wake rule and the gap's quantize.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::quantized::pinned_staging::PinnedBuf;
use candle::Device;
use candle_nn::kv_cache::{
    is_tier_standing_over_free, try_hold_chunk_locations, try_migrate_flight, ChunkedKvBacking,
    CompressionPolicy, HeldChunks, MigrateFlight, SealedSequence,
};
use crossbeam::channel::{self, Receiver, Sender};
#[cfg(feature = "memory-census")]
use heap_census::{enter, set_thread, Scope};

use super::cold_load::preallocate_pinned_scratch;
use super::flush::{FlushRequest, PassFlush};
use super::held_groups::{drain_in_held_groups, HeldDrain};
use super::resume::TurnChunkGrid;
use super::streams::StreamId;
use super::transfer::seal_to_chunk_images;
use super::warm_asks::WarmAsks;
use super::warm_ledger::WarmShare;
use crate::projection::{Conversation, MaintenanceOp, TurnKey};
use crate::scheduler::phase_ring::{
    migrate_sample, mix_due, push_idle_migrate, push_migrate, push_mix,
};
use crate::scheduler::relief_trace;
#[cfg(feature = "memory-census")]
use crate::store_gauges::gauge_if_due;
use crate::substrate::{ConvCompression, HotKv, ResidenceIndex, StoredSequence};
use sysinfo::System;

/// How often the loop wakes up on its own when no triggers arrive.
pub const DEFAULT_TICK: Duration = Duration::from_secs(5);

/// Minimum wall-clock between background **segment-maintenance** scans. The
/// tiering phases run every pass (one per trigger / tick), but the maintenance
/// candidate scan is O(live segments' records), so it's decoupled from the
/// trigger rate and run at most this often — reclaim is a slow background job,
/// not something that needs to react to every turn-seal.
pub const MAINTENANCE_INTERVAL: Duration = Duration::from_secs(15);

/// Absolute backstop on the number of hot→warm→cold passes the shutdown drain
/// runs. With the scheduler joined, the pending sets only shrink and the drain
/// converges in 1–2 passes; a persistently-failing turn (e.g. a repeated GPU
/// gather error) is caught FIRST by the loop's no-progress guard (it breaks after
/// one pass that fails to shrink the backlog). This cap only guards a pathological
/// shrink-then-regrow oscillation — the leftover re-ingests on the next reload.
const SHUTDOWN_DRAIN_MAX_PASSES: usize = 32;

/// The most pending residences one copy group takes: the bound that keeps the
/// migrate flight short enough for compaction to get in between groups.
const MAX_GROUP_RESIDENCES: usize = 8;

mod drive;
mod gauge;
mod handle;
mod pass;
#[cfg(test)]
mod tests_thread;
mod warm_copy;

use drive::*;
pub use gauge::*;
pub use handle::*;
pub(crate) use pass::*;
pub(crate) use warm_copy::*;
