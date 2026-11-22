//! A paged VRAM arena for the wide-Q provenance gallery, keeping each turn's folded
//! signatures resident on the GPU between reprojections.
//!
//! A turn's tokens occupy fixed-size pages, each one a slot in the device
//! reservation. An evicted or re-sealed turn releases only its own slots, and an
//! arena with nothing left in it hands its region back to the span. Residency is
//! keyed by stream, so every conversation on the card shares one arena, and each
//! turn records who holds it, so a conversation's turns can be given up when it
//! unloads. The substrate keeps the encoded signatures as the colder tier, so an
//! evicted turn is rebuilt on demand rather than kept on the card.
//!
//! - `accounting`: the bytes the gallery holds and can give back.
//! - `compaction`: packing pages toward the low end of the span.
//! - `evict`: giving up turns by recency and keep time.
//! - `eviction`, `keep_price`: the keep trade's rule, and its weight side's price.
//! - `held`: who holds a turn, and which tokens and groups it holds.
//! - `index_cache`: reusing a scan's index while its turns are unchanged.
//! - `reports`: the footprint, miss and address views.
//! - `residency`: making turns resident, uploading them when needed.
//! - `resident_turn`: a turn's pages and its bookkeeping.
//! - `setup`: construction and geometry.
//! - Tests: `tests_arena`.

mod batch_ensure;
mod belief_arena;
mod dedup;
pub(crate) mod eviction;
mod groups;
mod held_scan;
mod holder;
mod index_entries;
pub(crate) mod keep_price;
mod lease;
mod page_io;
mod pages;
mod probe_rows;
mod scan;
mod scan_plan;
pub mod scan_split;
mod token_stats;
mod token_votes;
mod turn_sigs;

pub use belief_arena::BeliefArena;
pub use dedup::{Codebook, Keep, KeepMask};
pub use groups::GroupSet;
pub use held_scan::{HeldScan, ScanAnswer, ScanReply};
pub use holder::{Holder, HolderKind};
pub use pages::{page_words, pages_for, transpose_to_pages, PAGE_TOKENS};
pub use scan::{PagedSegment, PagedWindow};
pub use scan_plan::ScanPlan;
pub use token_stats::TokenStats;
pub use token_votes::TokenVotes;
pub use turn_sigs::TurnSigs;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use std::result::Result as StdResult;

use arc_swap::ArcSwap;
use candle::{Device, Result};
use candle_nn::kv_cache::{
    arena_held_bytes, arena_regions, claim_arena_slots, plan_slot_moves, ArenaSlot, SlotTenant,
    SpanRegion,
};

use crate::persistence::streams::StreamId;

use dedup::Dedup;
use eviction::{index_cache_budget, index_cache_evictions, past_keep, preloads_on, Resident};
use lease::IndexLease;

/// How long a cached scan index stays once no scan stores or reuses it — a
/// reader that stopped scanning, not one between reprojections, which come
/// every 64 decoded tokens (`docs/gallery_residency.md` §5).
const INDEX_IDLE_MS: u64 = 60_000;

/// How often a reused index records its turns' use. What reads it — the keep
/// trade's `keep_within`, hours on every card measured — does not resolve five
/// seconds, and recording it is a walk of every turn under the residency lock.
const TOUCH_EVERY_MS: u64 = 5_000;

/// Paged VRAM gallery arena. Clone-free; share via `Arc<GalleryArena>`.
///
/// **Lock order:** `residency` → the slot arenas' own lock. A page claim and a
/// `PageRun` drop take the slot arenas' lock under `residency`; nothing in the slot
/// arenas ever calls back into the gallery, so the two never deadlock.
pub struct GalleryArena {
    residency: Mutex<HashMap<StreamId, ResidentTurn>>,
    lru_clock: AtomicU64,
    /// Bumped on every residency mutation (insert / evict / drop / move) — a
    /// count of churn, reported per scan. Not what validates a cached index:
    /// that is per turn ([`ResidentTurn::run_id`]), so an upload for one
    /// conversation does not invalidate another's index.
    residency_gen: AtomicU64,
    /// Source of [`ResidentTurn::run_id`]s.
    next_run_id: AtomicU64,
    /// `T*` in nanoseconds, `u64::MAX` for none — see [`Self::set_keep_within`].
    keep_within_ns: AtomicU64,
    /// Measured upload cost per byte, microseconds, as `f64` bits.
    upload_us_per_byte: AtomicU64,
    /// This arena's device tensor capabilities `(b1 BMMA, INT8 IMMA)`, queried
    /// once on first scan. Cached PER ARENA (not per process) so heterogeneous
    /// multi-GPU setups — e.g. mixed Ada/Blackwell — resolve each arena's
    /// backend ladder against its own device.
    tensor_caps: OnceLock<(bool, bool)>,
    /// Device bytes the cached indices may hold together
    /// ([`index_cache_budget`] of this arena's card).
    index_cache_bytes: u64,
    /// Whether this arena's card is large enough for the startup preload
    /// ([`eviction::preloads_on`]).
    preloads: bool,
    /// Per-scan indices (page_ptr / page entries / seg prefixes) keyed by segment
    /// fingerprint, reused when the same segment set is rescanned and every turn
    /// it references still holds the run it was built against — skipping the
    /// rebuild each reprojection. Keyed (not a single slot) so the several
    /// belief groups scanned per reprojection don't evict each other; an index
    /// goes when it is stale or idle past [`INDEX_IDLE_MS`], and, as a
    /// backstop, the least recently used when the cache is past
    /// [`Self::index_cache_bytes`].
    ///
    /// Validated per turn, not by a device-wide generation: with ingest sealing
    /// turns concurrently, a global generation moved on every scan, and a
    /// dialogue's 7.4M-token index was rebuilt from scratch each reprojection
    /// (1.8–2.2 s) for uploads that touched none of its turns. [`Self::changes`]
    /// only says when that check can be skipped; it never rebuilds an index.
    ///
    /// Written under this lock, which only a build or a stale index takes.
    /// Scans read [`Self::indexes`].
    index_cache: Mutex<HashMap<u64, Arc<scan::PagedIndex>>>,
    /// [`Self::index_cache`] as scans read it, without a lock, with the indices
    /// that left the cache while a scan might still hold them (`lease`).
    indexes: ArcSwap<Vec<(u64, Arc<scan::PagedIndex>)>>,
    /// Counts every change to a resident turn an index could name — evicted,
    /// dropped, moved or re-uploaded — ahead of the change. A new upload names
    /// no index's turn and does not count (`lease`).
    changes: AtomicU64,
    /// When the arena began, what [`scan::PagedIndex::touched_ms`] and
    /// [`scan::PagedIndex::used`] count from.
    born: Instant,
    /// One page of zeros: what a scan's index points a (page, group) at when
    /// the page does not hold that group — only ever a group the window's own
    /// segment weighs zero (`docs/gallery_residency.md` §4). Claimed on first
    /// need and never evicted.
    zero_page: Mutex<Option<ArenaSlot>>,
    /// The dedup overlay: the codebook and every turn's mask.
    dedup: Dedup,
    device: Device,
    wpt: usize,
    n_groups: usize,
    /// Words a token holds per fold group.
    gw: usize,
}

/// Weight of the newest upload in the arena's per-byte upload cost.
const UPLOAD_WEIGHT: f64 = 0.25;

mod accounting;
mod compaction;
mod evict;
mod held;
mod index_cache;
mod reports;
mod residency;
mod resident_turn;
mod setup;
#[cfg(test)]
mod tests_arena;

pub use compaction::*;
use held::*;
pub use reports::*;
use resident_turn::*;
#[cfg(test)]
use tests_arena::*;
