//! Expert cache bookkeeping for the MoE expert zone: slot management and the
//! score-based eviction policy.
//!
//! The cache tracks which experts are resident, which slot holds each one, and
//! the score each has earned. A batch's victims are ranked in one scan at the
//! row being served, and the caller decides for each one whether it may be
//! taken, so an expert whose row is still being computed keeps its slot. Victims
//! are taken behind the wave first, and the first MoE layers are never evicted.
//! Residents fall into classes by reload tier and by whether their row routed
//! them last time, and the classes run cheapest expected miss first. The
//! eviction score is a decaying recency frequency plus a slower decode-reuse
//! term, so the experts that decode keeps coming back to outlast a quiet step.
//!
//! - `bookkeeping`: slot and score tables, and the ranked victim scan.
//! - `policy`: the pinned depth, decay constants and the zone floor.
//! - `tests_distance`: tests of the order by distance behind the wave.
//! - `tests_floor`: tests of the zone floor and concessions.
//! - `tests_scores`: tests of score updates and protection.
//! - `tests_warm`: tests of the reload-tier passes.

use super::regret::class_index;
use super::resident::HostBacked;
#[cfg(feature = "tensor-assert")]
use super::slot_owners::SlotOwners;
use super::types::ExpertSlot;
use candle_nn::kv_cache::WeightZone;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

/// Mutable bookkeeping owned exclusively by the pipeline thread.
///
/// All fields are plain data — no `Arc`, no atomic types.
///
/// ## Slot lifecycle
///
/// Each slot is either free (in the zone's free list), or occupied (has an
/// `ExpertSlot` and a `slot_to_key` entry).  Occupied slots have a
/// `last_used` timestamp that determines eviction order.
///
/// ```text
/// Free:     slots[i] = None,  slot_to_key[i] = None
/// Occupied: slots[i] = Some(ExpertSlot), slot_to_key[i] = Some((moe, exp))
/// ```
///
/// ## Where a slot's bytes come from
///
/// The [`WeightZone`] owns the addresses. It is the right-hand side of the
/// device reservation, and it is also **the free list** — there is no second one
/// here. That matters for more than tidiness: the zone hands out the *rightmost*
/// free slot, which keeps live experts packed away from the boundary the KV side
/// pushes against, and a duplicate `Vec` free list here would have silently
/// undone that ordering on every eviction (`push` puts the freed index on top,
/// so the next load takes it back regardless of where it sits).
pub struct ExpertCacheInner {
    /// VRAM slots — created on-demand, indexed by slot_idx.
    /// **No Arc wrapping** — sole ownership.
    pub(crate) slots: Vec<Option<ExpertSlot>>,
    /// The addresses, and the free list over them.
    pub(crate) zone: WeightZone,
    /// Forward lookup: `(moe_layer_idx, expert_idx) -> slot_idx`.
    pub(crate) key_to_slot: HashMap<(usize, usize), usize>,
    /// Per-slot usage timestamp — higher = more recently used.
    /// Kept for recency tie-breaking within score-based eviction.
    pub(crate) last_used: Vec<u32>,
    /// Monotonically increasing counter, bumped on each cache access.
    pub(crate) generation: u32,
    /// Reverse map: `slot_idx -> (moe_layer_idx, expert_idx)` for eviction.
    pub(crate) slot_to_key: Vec<Option<(usize, usize)>>,

    // ── Score-based eviction state ──
    /// Flat score table: `expert_scores[layer * experts_per_layer + expert]`.
    /// A lightly-decayed access frequency — higher = more valuable = evicted last.
    pub(crate) expert_scores: Vec<f32>,
    /// Long-memory decode reuse, indexed like `expert_scores`: decode hits
    /// only, decayed by [`DECODE_REUSE_DECAY`] a pass. Weighted into
    /// [`Self::score`] by [`DECODE_REUSE_WEIGHT`].
    pub(crate) decode_reuse: Vec<f32>,
    /// Number of MoE layers (e.g. 48).
    pub(crate) num_moe_layers: usize,
    /// Experts per MoE layer (e.g. 128).
    pub(crate) experts_per_layer: usize,
    /// Early MoE layers exempt from eviction — [`pinned_layer_count`], fixed
    /// for the life of the cache and independent of capacity.
    ///
    /// It cannot track the boundary, because these are exactly the experts the
    /// pack and the warm tier do not store: unpinning one would leave it with
    /// nowhere to reload from. The zone's floor ([`minimum_resident_slots`]) is
    /// what guarantees the cache can always afford it.
    pub(crate) pinned_layers: usize,
    /// Per `(layer, expert)`: does a host copy hold it that a reload needs no
    /// pack read for — a resident pinned slot or a pageable warm copy
    /// (`resident::HostBacked`)? Shared with the stager, which labels the
    /// pinned slots: written at the startup fill and at pass boundaries.
    pub(crate) warm_backed: Arc<HostBacked>,
    /// Flat `layer * experts_per_layer + expert` → did this expert's row route
    /// it on the row's last visit? Rewritten per row as each is served
    /// ([`Self::mark_visit`]); the recency half of [`EVICTION_PASSES`].
    hit_last: Vec<bool>,
    /// The order the eviction passes run in: [`EVICTION_PASSES`] until
    /// [`Self::set_eviction_costs`] orders them by expected miss cost.
    passes: [(ReloadTier, bool); 4],
    /// The slot-tenancy tags bucketize's owner check reads, once a device
    /// cache has attached them ([`Self::attach_owners`]); every tenant a slot
    /// gains after that is written through.
    #[cfg(feature = "tensor-assert")]
    owners: Option<Arc<SlotOwners>>,
}

mod bookkeeping;
mod policy;
#[cfg(test)]
mod tests_distance;
#[cfg(test)]
mod tests_floor;
#[cfg(test)]
mod tests_scores;
#[cfg(test)]
mod tests_warm;

pub use policy::*;
