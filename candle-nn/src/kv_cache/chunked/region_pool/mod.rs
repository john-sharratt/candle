//! The device reservation that holds everything the engine owns on a card except the dense weights.
//!
//! One virtual span per device is claimed once and laid out from left to right:
//! a fixed block for the persistence staging and held ground, the dense weights,
//! the KV regions, the transient tier that exists only while a forward runs, and
//! the weight side's expert slots, filled from the right. The weight floor is the
//! boundary that moves, and it moves only between forwards, on the expert
//! pipeline thread. The KV side buys any ground it needs beyond its own regions
//! through the broker, and a span too small for the fixed blocks is refused.
//!
//! - `region_pool_impl`: the reservation, its layout and its pool.
//! - `reservations`: claiming each device's reservation on first use.
//! - `span_layout`: the reservation's current addresses and relocation plans.
//! - `span_sizing`: how large the reservation is from the card's free memory.
//! - `dense_block`: the dense weight block loaded before any region exists.
//! - `claim_region`: taking a region for an arena, sweeping and buying when short.
//! - `issue_region`: handing a region to a new owner, cleaning it first.
//! - `region_handle`: a claimed region, returned when its handle drops.
//! - `occupancy`: how the KV side's regions are used.
//! - `kv_negotiation`: offering spare ground to the weight side.
//! - `span_region`: regions held by tenants other than KV arenas.
//! - `transient_tier`: placing and releasing the tier for each forward.
//! - `weight_side`: the weight side's seller and the floor's moves.
//! - `tests_claims`, `tests_dirt`, `tests_held`, `tests_layout`, `tests_occupancy`: tests.

use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use candle::cuda_backend::cudarc::driver::result::memset_d8_async;
use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::cuda_backend::graph::{graph_budget_bytes, try_without_recording};
use candle::cuda_backend::install_held_ground;
use candle::cuda_backend::load_scratch;
use candle::vram::AllocClass;
use candle::Result;

use super::backing::{TIER_REFUSED_MARKER, TIER_SHORT_BY};
use super::bump_arena::KV_ARENA_MID_WAVE;
use super::chunk_ops::MIGRATION_STAGING_CAP_BYTES;
use super::growth_policy::{kv_grow_step, FreeRegionTarget, GrowthPolicy, Occupancy, Refusal};
use super::held_ground::{HeldClaim, HeldStats, SpanHeldGround, BOOTSTRAP_HELD_BYTES};
use super::held_ledger::HeldSite;
use super::reservation::Reservation;
use super::span_geometry::{blocked, ceiling_regions, claimable, tier_fits};
use super::types::TARGET_ARENA_BYTES;
use super::weight_zone::{INITIAL_KV_RESERVE, MIN_ELASTIC_RESERVE};

/// One region of the KV side. Every size class carves its arenas at this size.
pub const REGION_BYTES: usize = TARGET_ARENA_BYTES;

/// The widest the wave transient tier can ever be — the old fixed reservation,
/// kept only as the worst case the elastic middle must be able to *reach*.
///
/// Nothing reserves this any more. The tier is placed per forward at the width
/// that forward actually needs (`WavePlan::phase_bytes`), which for a
/// twenty-session decode is a few megabytes against the 912 MiB here. This
/// constant survives so [`MIN_ELASTIC_RESERVE`] can be checked against it: the
/// floor has to leave room for the widest wave, even though no wave but the
/// widest will use it.
const MAX_WAVE_TRANSIENT_BYTES: usize = super::wave_spans::WAVE_ATTN_BYTES
    + super::wave_spans::WAVE_FFN_BYTES
    + super::wave_spans::WAVE_FORWARD_BYTES;

/// The fixed block at the far **left** of the span: the held ground's bootstrap
/// ([`BOOTSTRAP_HELD_BYTES`]) first, then the persistence thread's staging.
///
/// Fixed for the process lifetime and out of both frontiers' way. Putting it
/// left rather than letting it float is what lets the weight boundary move
/// without ever having to reason about a copy stream that is not synchronised
/// to the wave — and what lets a model's load take held buffers before any region
/// exists. The held ground past the load is held arenas, span regions
/// (`held_ground.rs`).
///
/// **No model widens it.** The residual a window hands from one forward to the
/// next lives on the step's forward span or a creep's arena, sized to the rows
/// in flight and held only while they are (`docs/wave_step.md`); held here, it
/// would be priced at the widest forward the scheduler can send and stand on
/// every card for the life of the process.
const PERSIST_SPAN_BYTES: usize = BOOTSTRAP_HELD_BYTES + MIGRATION_STAGING_CAP_BYTES;

const _: () = assert!(
    MAX_WAVE_TRANSIENT_BYTES.is_multiple_of(REGION_BYTES)
        && PERSIST_SPAN_BYTES.is_multiple_of(REGION_BYTES),
    "every fixed block must be region-aligned so region carving stays exact"
);

// The floor is measured over the tradeable ground, `[region_base, span_end)`,
// which begins after the fixed block and the dense weights — so the fixed block
// is never drawn from it, and only the tier has to fit inside it.
const _: () = assert!(
    MIN_ELASTIC_RESERVE > MAX_WAVE_TRANSIENT_BYTES,
    "the floor must leave room for the widest wave's tier, or a forward cannot run"
);

/// The span when no governor has measured the card.
///
/// A fixed number rather than a fraction of what the driver reports free. The
/// fraction was tried and is wrong here: the only processes that get this far
/// without a governor are test binaries, which run hundreds of GPU tests
/// concurrently, so "half of free" depends on which tests happen to be in
/// flight when the first KV cache is built — and a run that lost the race
/// claimed nothing at all. This is large enough for the suite's peak arena
/// working set and small enough to leave the card to the tests themselves.
const TEST_SPAN_BYTES: usize =
    2 * 1024 * 1024 * 1024 + MAX_WAVE_TRANSIENT_BYTES + PERSIST_SPAN_BYTES;

struct RegionPool {
    reservation: Reservation,
    /// Device bytes in use just before this pool reserved its span — see
    /// [`RegionStats::pre_reservation_in_use_bytes`].
    pre_reservation_in_use: Option<u64>,
    /// First byte of the span.
    span_base: u64,
    /// Bytes of the span actually backed by physical memory. May be less than
    /// what was reserved, when the driver refused part-way.
    span_bytes: usize,
    /// The held ground at the span's base, for the memory report.
    held: Arc<SpanHeldGround>,
    /// First byte available to regions — past the persistence block and the
    /// dense weights. **Moves while the model loads, never after**: see
    /// [`claim_dense`].
    region_base: u64,
    /// First byte of the dense weight block, at the persistence block's right
    /// edge. Fixed for the pool's life.
    dense_base: u64,
    /// Bytes of the dense block handed out so far — the bump cursor.
    ///
    /// Grows only during model load and is frozen by the first region claim,
    /// because every region address is measured from this block's right edge.
    /// There is no free list and no release path: the weights live as long as
    /// the process does, and the reservation is torn down with it.
    dense_bytes: usize,
    /// The span was sized from a measured card, so it may host the model.
    ///
    /// **Without a VRAM governor the span is a fixed constant sized for KV
    /// alone**, and placing weights in it takes regions the KV side has no way
    /// to do without: Qwen3-8B's dense block swallowed a `TEST_SPAN_BYTES` span
    /// whole, leaving `0 regions` and a model that could not claim a single KV
    /// block. The pool is the right home for weights there, exactly as it was
    /// before any of this — so [`claim_dense`] refuses and
    /// `QMatMul::from_weights_with_mode` falls back, which is a path it already
    /// has for every other reason a destination can be unavailable.
    dense_capable: bool,
    /// The load phase is over: the dense block's right edge is final.
    ///
    /// Set by [`freeze_dense`], which is the boundary between loading and
    /// serving. Before it, the two load allocators own the span and no region
    /// exists; after it, the dense block is immovable and everything above it is
    /// runtime ground.
    ///
    /// A flag as well as the `next > 0` test because the two say different
    /// things. `next > 0` is "someone has already claimed, so it is too late" —
    /// a backstop that catches a caller who never froze. This is "loading is
    /// finished", which is knowable *before* the first region claim and is what
    /// makes a late dense claim an error at its own call site rather than a
    /// silent success followed by a corrupted arena much later.
    dense_frozen: bool,
    /// The leftmost byte the weight side occupies: **the moving boundary**.
    /// `span_end` until an expert cache installs a zone.
    weight_floor: u64,
    /// Regions that fit between [`Self::region_base`] and the transient tier:
    /// the KV side is `[0, total)`.
    total: usize,
    /// Lowest region index never yet handed out.
    next: usize,
    /// Returned regions, lowest first (principle 5: keep live data left-packed).
    free: BinaryHeap<Reverse<usize>>,
    live: usize,
    /// Of `live`, the regions a [`SpanRegion`] holds rather than an arena.
    ///
    /// Maintained by `SpanRegion`'s own construction and drop, which is the only
    /// place a span tenant's handle is minted, so it cannot drift from the handles
    /// that exist. What it is for is the fragmentation efficiency figure: without
    /// it, ground a recurrent state store legitimately holds is charged to the KV
    /// side as fragmentation nothing can pack away.
    span_live: usize,
    peak_live: usize,
    /// Per region, whether a live [`RegionHandle`] holds it right now.
    ///
    /// **The one thing `live`, `free` and `next` cannot tell you: whether a
    /// region has been handed to two tenants at once.** Those three are counts
    /// and a heap, and every double-issue keeps all of them self-consistent — a
    /// duplicate free-list entry pops twice, so `live` rises twice and `free`
    /// falls twice, and the arithmetic in `region_stats` still balances. The
    /// overlap is only visible per index, which is what this is.
    ///
    /// It matters more here than in an ordinary allocator because every address
    /// in the reservation is mapped: a second tenant handed a live region does
    /// not fault, it writes over the first one's data and the fault surfaces as
    /// a wrong number in whatever reads it next. The tenants are not alike
    /// either — a KV arena, the wave transient tier, and a sequence's recurrent
    /// store all claim from this list — so the reader of the corrupted bytes is
    /// usually in a different subsystem from the writer.
    ///
    /// Not behind `tensor-assert`: this is one indexed bool per claim and
    /// release, both of which already hold the pool mutex and one of which
    /// already does a device-wide quiesce. There is no fence and no allocation,
    /// so it costs nothing measurable, and a guard that is compiled out is not
    /// watching the production run where this actually happens.
    owned: Vec<bool>,
    /// Count of whole-device quiesces this pool has performed.
    ///
    /// A `cuCtxSynchronize` waits for every task on every stream of the context,
    /// so once it returns, **every** kernel launched before it has retired — not
    /// just the ones reading the region that triggered it. That makes one sync a
    /// blanket safety statement about all regions released before it, which is
    /// what lets the counter below turn a per-claim wait into a per-batch one.
    quiesce_epoch: u64,
    /// Per region, the epoch current when its bytes last belonged to someone
    /// else, and which streams may still be touching them ([`Dirt`]). **Epoch
    /// zero means never** — the region has been untouched since the
    /// reservation was mapped, so it is already zeroed and has no readers.
    ///
    /// A claim needs a *wait* only while the epoch still equals
    /// [`Self::quiesce_epoch`] — otherwise a quiesce has happened since and the
    /// region's readers are long gone — and only when the claim's own stream
    /// does not already order it behind them; but any non-zero epoch means it
    /// needs *cleaning*, because the last occupant's bytes are still there.
    ///
    /// # Three ways ground changes hands, and all three land here
    ///
    /// An arena releasing its region is the obvious one. The other two are the
    /// reason this is no longer called `released_epoch`: the **wave transient
    /// tier** stands on KV ground for the length of a forward and writes
    /// intermediates all over it, and the **weight boundary** hands over ground
    /// that was holding expert weights. A region arriving from either is exactly
    /// as dirty as a recycled one, and until this was written only the recycled
    /// path checked.
    dirt: Vec<Dirt>,
    /// Where the wave transient tier sits **while a forward is running**, and
    /// how big it is. `None` between forwards, when it occupies nothing.
    ///
    /// This is the one variable-size block in the span, and it is variable
    /// precisely because it vanishes: nothing is live in it at the moment its
    /// position or extent changes, so a resize leaves no hole and moves no data.
    /// That is why it sits *between* the arenas and the weights — the only place
    /// a block that changes size can absorb both sides' movement without one.
    transient_base: Option<u64>,
    transient_bytes: usize,
    /// The largest tier placed since boot, in bytes — **not** cleared on
    /// release.
    ///
    /// `transient_bytes` is live occupancy and reads zero between forwards,
    /// which is precisely when `spare_regions` runs: every caller of
    /// `reclaim_spare_ground` invokes it on the line after `end_wave_transient`.
    /// So the demand series never once contained the tier, and the weight side
    /// was free to take the ground the very next wave's tier needs — ground it
    /// cannot give back mid-forward, because `set_weight_floor` refuses while a
    /// wave generation is open.
    ///
    /// The old `total` seed masked this by making the first two windows report
    /// no spare at all. Removing the seed exposed it, and a geometric grow step
    /// made it reachable within a few waves rather than dozens.
    transient_high_water: usize,
    /// The tiers the last [`RECENT_FORWARDS`] forwards placed, in bytes, as a ring
    /// written at [`Self::recent_at`] — **not** cleared on release. The growth
    /// negotiation leaves the widest of them free (see
    /// [`Occupancy::recent_tier_bytes`]).
    recent_tiers: [usize; RECENT_FORWARDS],
    /// Where the next forward's tier is written in [`Self::recent_tiers`].
    recent_at: usize,
    /// Whether prompts are waiting to be prefilled, as the scheduler last said
    /// ([`set_prompts_waiting`]).
    prompts_waiting: bool,
    /// The widest tier placed since prompts began waiting; zero while none do.
    /// The growth negotiation leaves it free (see
    /// [`Occupancy::waiting_tier_bytes`]): between two prefills the last forward
    /// is a decode, and the next prefill needs this ground back.
    waiting_tier_bytes: usize,
    /// The largest tier placed since [`span_partition`] last read it — the
    /// wave's activation ground, which a reading taken between forwards would
    /// otherwise always see as zero.
    tier_peak: usize,
    /// Tiers placed since the pool was made — one per forward: the clock a
    /// per-wave reading of the span is taken against.
    tiers_placed: u64,
    /// Bytes of the fixed left block carved so far: the held ground, then the
    /// persistence staging.
    persist_carved: usize,
    /// The whole fixed left block — [`PERSIST_SPAN_BYTES`].
    persist_bytes: usize,
    /// The held ground's bootstrap block at its base — [`BOOTSTRAP_HELD_BYTES`].
    held_bytes: usize,
    /// Fresh regions claimed while a wave's transient tier was placed.
    ///
    /// **A tripwire that must read zero.** A region claim creates an arena, and
    /// an arena may only be created between forwards —
    /// `BackingInner::arena_window` blocks until then. A non-zero reading names
    /// a creation path that skipped that gate and moved the arena frontier under
    /// a tier already placed against it.
    ///
    /// It used to be a *measurement*: the count of claims that would have to fit
    /// in a gap above the frontier for the tier to be anchored there, which on
    /// the quantized path ran into the hundreds because the compressor creates
    /// size-class arenas as it chooses formats. That number was read as proof
    /// the frontier anchor was impossible. It was proof the sealing thread was
    /// allocating arenas inside the wave.
    fresh_claims_during_wave: usize,
    /// Region claims refused because the tier's base was the ceiling.
    ///
    /// The same tripwire from the other side: a claim that ran out of ground
    /// *with a tier standing* is a claim that arrived inside a wave. A claim
    /// that runs out with no tier standing is ordinary pressure and buys more
    /// (`set_ground_broker`).
    refusals_during_wave: usize,
    /// Whether the weight side may take ground, and the state that decision
    /// carries between negotiations.
    ///
    /// **The policy, kept out of the pool on purpose.** This module owns the
    /// reservation, the free list and the tier — the things that need a device.
    /// Which of them add up to "spare" is arithmetic, and
    /// [`super::growth_policy`] holds it so a trajectory can be *run* without
    /// one: every partition defect so far has been a sequence of individually
    /// defensible answers walking the boundary somewhere bad, which no
    /// single-call test can see. Same split, and the same reason, as
    /// [`super::weight_zone`] on the mirror side.
    growth: GrowthPolicy,
    /// The free regions the KV side's scheduler defends ([`set_kv_free_target`]),
    /// which the growth policy leaves standing. None until a scheduler states one.
    kv_target: FreeRegionTarget,
}

mod claim_region;
mod dense_block;
mod issue_region;
mod kv_negotiation;
mod occupancy;
mod region_handle;
mod region_pool_impl;
mod reservations;
mod span_layout;
mod span_region;
mod span_sizing;
#[cfg(test)]
mod tests_claims;
#[cfg(test)]
mod tests_dirt;
#[cfg(test)]
mod tests_held;
#[cfg(test)]
mod tests_layout;
#[cfg(test)]
mod tests_occupancy;
mod transient_tier;
mod weight_side;

pub(crate) use claim_region::*;
pub use dense_block::*;
pub use issue_region::*;
pub use kv_negotiation::*;
pub use occupancy::*;
pub(crate) use region_handle::*;
pub use reservations::*;
pub use span_layout::*;
pub use span_region::*;
pub(crate) use span_sizing::*;
pub use transient_tier::*;
pub use weight_side::*;
