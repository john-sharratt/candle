//! Expert cache bookkeeping for the MoE expert zone: slot management and the
//! score-based eviction policy.
//!
//! The cache tracks which experts are resident, which slot holds each one, and
//! the score each has earned. A batch's victims are ranked in one scan at the
//! row being served, and the caller decides for each one whether it may be
//! taken, so an expert whose row is still being computed keeps its slot. Victims
//! are taken behind the wave first, and the first MoE layers are never evicted.
//! The experts their row did not route last time go before the ones it did, and
//! within each group every resident competes on one key: least used first, with
//! a pack-only expert's small reload-cost protection blended in rather than run
//! as a pass of its own. The eviction score is a decaying recency frequency plus
//! a slower decode-reuse term, so the experts that decode keeps coming back to
//! outlast a quiet step.
//!
//! - `bookkeeping`: slot and score tables, and the ranked victim scan.
//! - `policy`: the pinned depth, decay constants and the zone floor.
//! - `tests_distance`: tests of the order by distance behind the wave.
//! - `tests_floor`: tests of the zone floor and concessions.
//! - `tests_scores`: tests of score updates and protection.
//! - `tests_warm`: tests of the reload-tier blend.

use super::card_key::{CardModels, ExpertKey, ModelIx};
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
/// Occupied: slots[i] = Some(ExpertSlot), slot_to_key[i] = Some((model, row, expert))
/// ```
///
/// Every expert is named by its [`ExpertKey`], `(model, row, expert)`, so the
/// models a card registers share one zone, one set of slots and one victim scan
/// (`docs/parasite_model.md` §11.2). The flat score tables lie one model after
/// another, each at its own base ([`CardModels`]).
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
    /// Forward lookup: `(model, row, expert) -> slot_idx`.
    pub(crate) key_to_slot: HashMap<ExpertKey, usize>,
    /// Per-slot usage timestamp — higher = more recently used.
    /// Kept for recency tie-breaking within score-based eviction.
    pub(crate) last_used: Vec<u32>,
    /// Monotonically increasing counter, bumped on each cache access.
    pub(crate) generation: u32,
    /// Reverse map: `slot_idx -> (model, row, expert)` for eviction.
    pub(crate) slot_to_key: Vec<Option<ExpertKey>>,

    // ── Score-based eviction state ──
    /// The models registered on the card and where each one's experts lie in
    /// the flat tables below.
    pub(crate) models: CardModels,
    /// Flat score table, one entry per expert of every model
    /// ([`CardModels::flat`]). A lightly-decayed access frequency — higher =
    /// more valuable = evicted last.
    pub(crate) expert_scores: Vec<f32>,
    /// Long-memory decode reuse, indexed like `expert_scores`: decode hits
    /// only, decayed by [`DECODE_REUSE_DECAY`] a pass. Weighted into
    /// [`Self::score`] by [`DECODE_REUSE_WEIGHT`].
    pub(crate) decode_reuse: Vec<f32>,
    /// Each model's early MoE rows exempt from eviction — [`pinned_layer_count`]
    /// of its rows, fixed for the life of the cache and independent of
    /// capacity. Indexed by model.
    ///
    /// It cannot track the boundary, because these are exactly the experts the
    /// pack and the warm tier do not store: unpinning one would leave it with
    /// nowhere to reload from. The zone's floor ([`minimum_resident_slots`]) is
    /// what guarantees the cache can always afford it.
    pinned_rows: Vec<usize>,
    /// Per model, per `(row, expert)`: does a host copy hold it that a reload
    /// needs no pack read for — a resident pinned slot or a pageable warm copy
    /// (`resident::HostBacked`)? Indexed by model; each is shared with that
    /// model's stager, which labels its pinned slots: written at the startup
    /// fill and at pass boundaries.
    pub(crate) warm_backed: Vec<Arc<HostBacked>>,
    /// Per expert, flat like `expert_scores` → did this expert's row route it
    /// on the row's last visit? Rewritten per row as each is served
    /// ([`Self::mark_visit`]); what [`RECENCY_PASSES`] splits residents by.
    hit_last: Vec<bool>,
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
