//! Expert cache: four tiers and one device-side expert path.
//!
//! The infrastructure for MoE (Mixture-of-Experts) models whose experts do not
//! all fit in VRAM. Every expert is readable by the GPU at all times, from VRAM
//! (a weight-zone slot), from pinned host memory (a warm-tier or pad slot), or
//! — cold — from the NVMe pack once the stager has staged it into the pad. The
//! live table (`live_table`) names where; `docs/moe_live_dispatch_design.md`
//! §0 is the design.
//!
//! ## Architecture
//!
//! - **The forward thread** (`dispatch`) enqueues each MoE layer — bucketize,
//!   gather, gate / up / down, scatter — and never waits on the host. The expert
//!   GEMMs' worker blocks copy every non-VRAM expert's row tiles into VRAM
//!   scratch and compute them; only a cold expert's workers wait, on a host
//!   store.
//! - **The stager** (`stager`) is the only reader of the pack: it stages each
//!   routed row's cold experts into the pad and publishes them, with no CUDA
//!   call on the way.
//! - **The pipeline thread** (`pipeline`) owns VRAM residency — scores, the
//!   Markov transition matrix, promotion into VRAM with the copy engine,
//!   eviction — off the critical path. It exclusively owns
//!   `ExpertCacheInner` with `&mut self`; the per-expert places it shares with
//!   the stager live behind one lock (`residency`), which also writes the
//!   table, under the reclaim rule (`reclaim`).
//!
//! ## Eviction policy
//!
//! Eviction is a **retarget** — every expert has a copy in the pack, and the
//! warm tier is immutable — so it moves no bytes. A promotion takes a free
//! slot, else the best victim [`cache::ExpertCacheInner::rank_victims`] offers
//! that the reclaim rule lets go now: a row with no invocation in flight.
//!
//! ### The eviction key: `score × position`, run once per reload tier
//!
//!   - **score** — the lightly-decayed access frequency above; the dominant
//!     term, so the cache behaves as LFU with a recency decay.
//!   - **position** — a mild `[0.5, 1.0]` multiplier that FALLS with forward
//!     (wrapped) reuse distance.  Bélády's direction, and computable rather
//!     than predicted: the layer traversal is a cycle, so the distance to an
//!     expert's next use is a subtraction.  The layer about to be routed is
//!     most protected; the layer just executed is the preferred victim.
//!   - **reload tier** — every eviction runs that policy over the experts the
//!     warm tier holds first, and over the NVMe-pack-only experts only for
//!     the shortfall.  The warm tier and the pack hold disjoint experts and
//!     routing is trained balanced, so which experts VRAM holds does not move
//!     the hit rate — it decides where the misses land, and a warm miss is a
//!     host-to-device copy where a pack miss is a page-cache-bypassing NVMe
//!     read an order of magnitude slower.
//!
//! ### Early-layer pinning
//!
//! Experts in the first [`PINNED_LAYERS`] MoE layers are never evicted: they
//! run first every pass with no compute ahead of them to overlap a DMA
//! against, so evicting them guarantees a cold miss at maximum stall.
//!
//! The depth is a **constant**, not a function of capacity, because these are
//! also the experts that have no copy anywhere else: [`pack`] writes no record
//! for them and the warm tier's draw skips them, so a cache that unpinned one
//! under pressure would strand it with nowhere to reload from.  What guarantees
//! the cache can always pay for them is the zone's floor —
//! `cache::minimum_resident_slots` prices the pinned set plus a full working
//! layer, and the elastic boundary may not retract below it.
//!
//! Dropping their host and disk copies is the point, not a side effect: on the
//! 3.6-35B it returns 943 MiB of pinned host RAM to the evictable set — the
//! only set that generates misses — and the same again on disk.
//!
//! ### Windowed prefetch eviction
//!
//! Speculative promotion takes free slots first, and may make room only from
//! the **furthest-behind** layers (`cache::PREFETCH_EVICT_WINDOW`), never the
//! wrapped tail of the pass. Near-future layers are structurally out of its
//! reach, so a mispredicted prefetch cannot displace an expert this sweep is
//! about to need.
//!
//! ## Transition matrix and speculative prefetch
//!
//! An online-learned transition matrix tracks expert→expert routing
//! patterns across adjacent MoE layers.  For each pair of consecutive
//! layers `(L, L+1)`, a `[E × E]` co-occurrence matrix records how often
//! an expert at layer L is followed by each expert at layer L+1.
//!
//! The matrix is built incrementally during inference — no calibration pass
//! required, and no extra compute: it consumes routing IDs only, never live
//! activations, so it is free to evaluate and shared across every token in a
//! wave that routed to the same expert.  At each layer the predictor ranks the
//! likely *non-cached* experts for the next layer: those with a pinned copy are
//! promoted while the current layer computes, cold ones are staged into the pad.
//!
//! The fan-out is **not** a fixed top-`K`.  Each candidate must clear a
//! per-source relative confidence floor, ranked by pointwise mutual
//! information and capped at a fixed maximum, so depth tracks demand
//! *diversity* rather than demand *width* — see [`transition`] for why the cap
//! must not scale with the batch.
//!
//! Correct predictions turn misses into hits.  Incorrect ones occupy a slot the
//! windowed eviction above will reclaim, taken from layers the wave has already
//! left.
//!
//! Prediction is worth nothing at prefill width, where the next layer routes to
//! most of its experts and there is nothing to guess: a prefill-width row
//! promotes the next row's scored experts instead.
//!
//! ## Module structure
//!
//! | File | Contents |
//! |------|----------|
//! | [`types`]      | Shared data types (`MmapExpertRef`, `ExpertSlot`, stats, messages) |
//! | [`cache`]      | `ExpertCacheInner` — VRAM slots and the eviction policy |
//! | [`compute`]    | `QMatMul` re-export |
//! | [`transition`] | `TransitionMatrix` — online-learned routing predictor |
//! | [`pack`]       | `ExpertPack` — the authoritative cold tier on disk |
//! | [`pinned`]     | `WarmPool` — pinned host memory slots, and the warm draw |
//! | [`warm_tier`]  | `WarmTier` — the warm tier: pinned up to the page-lock ceiling, pageable past it |
//! | `pad`          | `Pad` — the mutable pinned tier cold experts are staged into |
//! | [`page_pressure`] | make the OS release RAM before a page-lock retry |
//! | `live_table`   | the live `[3][rows][E]` table in mapped memory, written by host stores |
//! | `residency`    | where every expert's copies are; the lock that writes the table |
//! | `reclaim`      | tickets, and when a slot may be reused or an entry go to 0 |
//! | `dispatch`     | the device-side expert forward on the forward thread |
//! | `stager`       | the stager thread: pack and pageable reads into the pad |
//! | [`pipeline`]   | the pipeline thread: promotion, prefetch, scores |
//! | `promo`        | the promotion ring the GPU fills missed experts into |
//! | `boundary`     | the elastic weight/KV boundary move |
//! | `slot_image`   | one expert's slot image: offsets, views, uploads |
//! | `startup`      | building or reusing the pack, and the startup fill |
//! | [`handle`]     | `ExpertCache` public API |

#[cfg(feature = "cuda")]
mod boundary;
mod cache;
pub(crate) mod compute;
/// Copy-engine promotions, issued off the pipeline thread.
#[cfg(feature = "cuda")]
mod copier;
#[cfg(feature = "cuda")]
mod dispatch;
#[cfg(test)]
mod eval;
/// `pub(crate)` so the layer warm tier can size itself through the same three
/// host-RAM ceilings this one does — see `handle::warm_slots_for`.
pub(crate) mod handle;
#[cfg(feature = "cuda")]
mod live_table;
#[cfg(all(test, feature = "cuda"))]
mod matmul_baseline;
/// `pub(crate)` so the layer pack shares this one's repack fingerprint rather
/// than growing a second definition of the same sweep — the two packs hold
/// weights repacked by identical code, so a change that invalidates one must
/// invalidate the other.
pub(crate) mod pack;
#[cfg(feature = "cuda")]
mod pad;
#[cfg(feature = "cuda")]
mod page_pressure;
/// `pub(crate)` for [`WarmPool`](pinned::WarmPool) alone, which is a generic
/// pinned-slot allocator with nothing expert-specific in it and is shared with
/// [`layer_stream`](crate::models::layer_stream). Its neighbour
/// `stratified_membership` *is* expert-specific — see
/// `layer_stream::warm` on why a layer tier draws a contiguous run instead.
pub(crate) mod pinned;
#[cfg(feature = "cuda")]
mod pipeline;
#[cfg(feature = "cuda")]
mod promo;
#[cfg(feature = "cuda")]
mod reclaim;
#[cfg(feature = "cuda")]
mod residency;
#[cfg(feature = "cuda")]
mod slot_image;
/// Fletcher-32 fingerprints of the resident expert weights, taken once after
/// the fill so a later corruption can be told from a bad fill.
#[cfg(feature = "cuda")]
pub mod slot_integrity;
/// The slot-tenancy tags bucketize's owner check reads.
#[cfg(feature = "tensor-assert")]
mod slot_owners;
#[cfg(feature = "cuda")]
mod stager;
/// Which invocation of each row the device has begun — the reclaim key.
#[cfg(feature = "cuda")]
mod started;
#[cfg(feature = "cuda")]
mod startup;
mod transition;
mod types;
#[cfg(feature = "cuda")]
pub(crate) mod warm_tier;
mod weight_plan;

// Re-exports — the public API of this module.
pub use crate::models::profile::ProfileSnapshot;
#[cfg(feature = "cuda")]
pub use boundary::grow_tally;
#[cfg(feature = "cuda")]
pub use cache::minimum_resident_slots;
/// Shared with the layer cache, which pins the same count for the same reason:
/// the leading layers are reached first on every forward and have the least
/// time to be fetched, so they are the ones worth never fetching at all.
pub use cache::PINNED_LAYERS;
pub use handle::ExpertCache;
pub use handle::ExpertCacheSetup;
#[cfg(feature = "cuda")]
pub use handle::{
    last_warm_tier_sizing, WarmTierSizing, CEILING_AVAILABLE, CEILING_HOST_BUDGET, CEILING_NONE,
    CEILING_PINNABLE, WARM_TIER_HEADROOM,
};
/// What the weight zone must be carved into to hold one expert.
///
/// The model loader needs this **before** the cache exists: the zone's slot size
/// decides its capacity, its capacity decides where the weight boundary sits,
/// and the boundary has to be placed before a single expert is uploaded into it.
#[cfg(feature = "cuda")]
pub(crate) use pinned::layer_geometries;
#[cfg(feature = "cuda")]
pub(crate) use slot_image::slot_bytes_for;
pub use types::{ExpertSlot, MmapExpertRef, PipelineStats};
pub use weight_plan::{WeightPlan, WeightPlanning};
