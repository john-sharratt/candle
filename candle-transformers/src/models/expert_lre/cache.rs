//! Expert cache bookkeeping — slot management, eviction policy, score-based.
//!
//! This module contains [`ExpertCacheInner`], the mutable bookkeeping
//! structure that tracks which experts are resident in VRAM, manages
//! slot allocation, and implements the score-based eviction policy.
//!
//! ## Eviction policy
//!
//! Frequency-dominated, layer-aware, with pinning.  In brief:
//!
//! 1. **Exact-demand batch eviction** — classify counts a layer's misses
//!    before any load and evicts exactly `misses − free` bottom-scored slots
//!    in one scan ([`ExpertCacheInner::demand_eviction`]), scored at the
//!    wave's real layer with the layer's own hits protected. Eviction is a
//!    pure drop (the cold pack holds every expert; the warm tier is
//!    immutable), so there is no copy to hide and nothing to do ahead of
//!    time.
//! 2. **Layer-aware forced eviction** — the per-miss backstop when the batch
//!    scan could not free enough (pathological): prefer evicting a low-scored
//!    expert from a layer already executed this pass (behind the wave, so it
//!    can never cascade), then fall back to the global lowest-scored victim.
//! 3. **Early-layer pinning** — the first [`PINNED_LAYERS`] layers are never
//!    evicted (they run first every pass with no compute to hide a reload).
//! 4. **Windowed prefetch eviction** — speculative prefetch makes room only
//!    from the furthest-behind layers ([`PREFETCH_EVICT_WINDOW`]).
//! 5. **Reload-tier passes** — each of the above runs over the warm-backed
//!    experts first and over the NVMe-pack-only experts only for what that
//!    pass could not free ([`EVICTION_PASSES`]). With no warm tier, or with
//!    one holding everything, there is one class and the policy is unchanged.
//!
//! ## Score table
//!
//! A flat `Vec<f32>` indexed by `layer * experts_per_layer + expert` records a
//! lightly-decayed access frequency: higher = more valuable = evicted last.
//! Updated by pipeline events:
//!
//! - **Cache hit**: +1.0
//! - **Prediction hit**: +0.3 (a speculative load the layer actually routed to)
//! - **End-of-pass decay**: ×0.85 (recency-weighting of the frequency)

use super::types::ExpertSlot;
use super::zone_geometry::ZoneGeometry;
use candle::Result;
use candle_nn::kv_cache::WeightZone;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

/// Number of early MoE layers whose experts are never evicted.
///
/// These layers run first every pass and have zero compute to overlap
/// with DMA — evicting them guarantees cold misses with maximum stall.
/// A single decode step routes top-8 per layer, so pinning layers 0–1 holds
/// ~16 experts; a batch wide enough to route everywhere holds all of them, which
/// is what [`minimum_resident_slots`] prices.
///
/// # Why this is a constant and not a function of capacity
///
/// It used to be derived from the zone's live capacity, so a small card
/// degraded to less pinning rather than to a cache that could not evict. That
/// is no longer available, because the pinned set is now **the set of experts
/// with no host or disk copy at all**: a permanently-resident expert is never
/// reloaded, so [`pack`](super::pack) omits its record and the warm tier omits
/// it from the draw. A pinned count that varied with capacity would make a
/// pack built on one machine unreadable on another — the second machine would
/// unpin a layer whose records were never written and then fail every load
/// against it.
///
/// So the depth is fixed, and the *floor* is what guarantees it can be paid:
/// [`minimum_resident_slots`] prices the pinned set plus a full working layer,
/// and the zone may never retract below it.
pub const PINNED_LAYERS: usize = 2;

/// How many layers this model actually pins.
///
/// [`PINNED_LAYERS`], except for a model with fewer MoE layers than that — in
/// which case every layer is pinned and there is no evictable set, the
/// all-resident case the cache already handles inline.
///
/// Both the pack writer and the warm-tier draw derive their skip from this, so
/// a single number decides which experts have a reload path.
pub(crate) fn pinned_layer_count(num_moe_layers: usize) -> usize {
    PINNED_LAYERS.min(num_moe_layers)
}

/// The fewest slots the cache can serve a token with, for a model with
/// `experts_per_layer` experts in each MoE layer.
///
/// **The eviction scan cannot touch layers `0..PINNED_LAYERS`.** A batch wide
/// enough to route to every expert in those layers fills
/// `PINNED_LAYERS × experts_per_layer` slots that no victim search will ever
/// select. Give the zone fewer slots than that and it can reach a state where
/// every resident slot holds a pinned-layer expert: the `layer >= PINNED_LAYERS`
/// filter in [`ExpertCacheInner::evict_lru_for`] matches nothing, and every load
/// from then on fails with "Expert cache full, cannot evict (all pinned)" — for
/// the life of the process, because nothing in that state can ever free a slot.
///
/// The daemon reached it: the boundary retracted to 297 slots against a
/// pinned-eligible set of 384, and the next 1,774 wave steps all failed
/// identically. This is the number that must never be crossed, and since the
/// boundary is otherwise free to trade expert residency for KV ground on demand,
/// it is the **only** thing standing between a hungry KV side and a dead engine.
///
/// Priced as the pinned set **plus a full working layer, plus one**, because
/// [`PINNED_LAYERS`] is fixed: nothing sheds pinned layers under pressure, so
/// this floor is all that keeps the eviction scan supplied with candidates. A
/// zone sitting exactly on it holds every pinned expert, a worst-case routed set
/// for the layer executing, and one slot for the load that triggered the
/// eviction to land in.
///
/// The arithmetic is unchanged from when pinning was three capacity-derived
/// layers and this priced only `3 × n + 1`: two pinned layers plus a working
/// layer is the same `3 × n`. So no model's boundary placement moves — which
/// matters, because this number feeds the opening size of every MoE model and
/// raising it starves the KV side (Qwen3-30B OOMs at load).
pub fn minimum_resident_slots(experts_per_layer: usize) -> usize {
    (PINNED_LAYERS + 1) * experts_per_layer + 1
}

/// How many of the furthest (just-behind, wrapping) layers are eligible as
/// prefetch make-room victims. Caps how far back eviction reaches from the
/// current layer (`current-1 .. current-PREFETCH_EVICT_WINDOW`), keeping it off
/// the near-future layers about to be used. At the pinned boundary this window
/// lands on the wave's tail. See [`ExpertCacheInner::evict_for_prefetch_batch`].
#[cfg(any(feature = "cuda", test))]
pub(crate) const PREFETCH_EVICT_WINDOW: usize = 5;

/// Where an evicted expert's next miss is served from.
///
/// The warm tier and the NVMe pack hold disjoint experts, and the model's
/// routing is trained balanced, so which experts VRAM holds barely moves its
/// hit rate — it decides where the misses land. Every slot VRAM spends on a
/// warm-backed expert leaves one more pack-only expert to miss on disk (a
/// 2.9 MB page-cache-bypassing read near a millisecond, against ~116 µs H2D
/// from pinned host memory). So every eviction runs its policy in
/// [`EVICTION_PASSES`] order: over the warm-backed experts first, and over the
/// pack-only experts only for whatever the first pass could not free.
///
/// Measured on Flash-Next at 16 GB against the single-pass policy: pack share
/// of loads 68 % → 61 / 63 / 50 % (BF16×1 / BF16×8 / C5×2), decode 16.2 /
/// 72.1 / 30.1 → 16.2 / 74.4 / 32.1 t/s, for a hit rate 2–3 points lower.
/// Protecting the hottest 20 % or 40 % of residents from the first pass
/// recovered under a point of that hit rate, gave back the pack saving about
/// as fast, and decoded no faster — the frequency signal is spread across the
/// warm-backed experts, not concentrated at the top, and a warm miss is cheap
/// enough that the extra ones cost less than the pack reads they replace.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ReloadTier {
    /// A warm (pinned host) copy exists — the cheap reload.
    Warm,
    /// Only the NVMe pack holds it.
    Nvme,
}

/// The order eviction passes visit the reload tiers in.
const EVICTION_PASSES: [ReloadTier; 2] = [ReloadTier::Warm, ReloadTier::Nvme];

/// One eviction candidate as the batch scans rank it within a pass.
#[derive(Clone, Copy)]
struct VictimKey {
    slot: usize,
    /// Frequency, with whatever positional factor the scan applies.
    score: f32,
    /// Wrapped forward distance from the current layer.
    dist: usize,
    lru: u32,
}

impl VictimKey {
    /// Best victim first: lowest score, then farthest, then least recently
    /// used.
    fn order(a: &Self, b: &Self) -> Ordering {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(Ordering::Equal)
            .then(b.dist.cmp(&a.dist))
            .then(a.lru.cmp(&b.lru))
    }
}

/// Mutable bookkeeping owned exclusively by the pipeline thread (threaded
/// mode) or the Mutex (inline mode).
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
    /// The zone's live shape, published for readers that do not own the cache.
    ///
    /// **This is what invalidates the GPU dispatch tables, and comparing
    /// capacities is not enough to replace it.** The tables capture a raw
    /// address per expert once, on the reasoning that an all-resident cache's
    /// weights never move. A concession breaks that: the slots at the frontier
    /// are evicted or relocated so the wave transient tier can stand on their
    /// ground, and the tier then writes activations over them.
    ///
    /// The reason a capacity check misses it is that the zone GROWS BACK. By
    /// the time anything looks, capacity and floor read exactly as they did at
    /// load — while the slots that were conceded now hold a different expert,
    /// or nothing, and the tables still name them. A monotonic count of
    /// concessions cannot be undone by a regrow, which is the whole point.
    ///
    /// Shared rather than a plain field because the reader is the *handle*, on
    /// the forward thread, while this struct is moved into the pipeline
    /// thread's state — see [`ZoneGeometry`] for the dead-code bug that cost.
    pub(crate) geometry: Arc<ZoneGeometry>,
    /// Reverse map: `slot_idx -> (moe_layer_idx, expert_idx)` for eviction.
    pub(crate) slot_to_key: Vec<Option<(usize, usize)>>,

    // ── Score-based eviction state ──
    /// Flat score table: `expert_scores[layer * experts_per_layer + expert]`.
    /// A lightly-decayed access frequency — higher = more valuable = evicted last.
    pub(crate) expert_scores: Vec<f32>,
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
    /// Flat `layer * experts_per_layer + expert` → does a warm (pinned host)
    /// copy of this expert exist?
    ///
    /// Written once when the warm tier is filled and never again — the tier is
    /// immutable, which is what lets the eviction policy treat this as a
    /// property of the expert rather than something to re-check.
    pub(crate) warm_backed: Vec<bool>,
}

impl ExpertCacheInner {
    /// Create a new empty cache over `zone`'s slots.
    ///
    /// * `num_moe_layers` — total MoE layers (e.g. 48)
    /// * `experts_per_layer` — experts per layer (e.g. 128)
    pub(crate) fn new(zone: WeightZone, num_moe_layers: usize, experts_per_layer: usize) -> Self {
        let num_slots = zone.capacity();
        let geometry = Arc::new(ZoneGeometry::new(
            num_slots,
            zone.slot_base(num_slots.saturating_sub(1)),
        ));
        // Publish this grid's slot extents to the between-waves span audit.
        //
        // Through the shared `ZoneGeometry` rather than through `self`, for the
        // same reason every other out-of-cache reader goes that way: the audit
        // runs on the forward thread while the cache is owned elsewhere. The
        // closure re-reads on every call because all of it moves — a concession
        // changes `capacity` and `frontier` together, and a cached pair would
        // describe the grid as it was before the boundary moved, which is
        // precisely the state this is meant to catch.
        //
        // `span_end` is reconstructed rather than stored: `frontier` is
        // `slot_base(capacity - 1)` = `span_end - capacity * slot_bytes`, so the
        // top of the span follows from the two published numbers and the slot
        // size.
        #[cfg(feature = "tensor-assert")]
        {
            let g = geometry.clone();
            let slot_bytes = zone.slot_bytes();
            super::span_claims::register(move || {
                let capacity = g.capacity();
                if capacity == 0 || slot_bytes == 0 {
                    return None;
                }
                let span_end = g.frontier() + (capacity * slot_bytes) as u64;
                Some((span_end, slot_bytes, capacity))
            });
        }
        Self {
            slots: (0..num_slots).map(|_| None).collect(),
            zone,
            key_to_slot: HashMap::new(),
            last_used: vec![0u32; num_slots],
            generation: 0,
            geometry,
            slot_to_key: vec![None; num_slots],
            expert_scores: vec![0.0f32; num_moe_layers * experts_per_layer],
            num_moe_layers,
            experts_per_layer,
            pinned_layers: pinned_layer_count(num_moe_layers),
            warm_backed: vec![false; num_moe_layers * experts_per_layer],
        }
    }

    /// Record which experts the warm tier holds, once its fill is decided.
    ///
    /// Called before the first forward and never again.
    pub(crate) fn set_warm_backed(&mut self, membership: &[(usize, usize)]) {
        self.warm_backed.iter_mut().for_each(|b| *b = false);
        for &(layer, expert) in membership {
            let idx = layer * self.experts_per_layer + expert;
            if idx < self.warm_backed.len() {
                self.warm_backed[idx] = true;
            }
        }
    }

    /// Where `(layer, expert)` reloads from after an eviction.
    #[inline]
    fn reload_tier(&self, layer: usize, expert: usize) -> ReloadTier {
        let idx = layer * self.experts_per_layer + expert;
        if self.warm_backed.get(idx).copied().unwrap_or(false) {
            ReloadTier::Warm
        } else {
            ReloadTier::Nvme
        }
    }

    /// Slots that exist — the zone's capacity, which is also `slots.len()`.
    pub(crate) fn num_slots(&self) -> usize {
        self.zone.capacity()
    }

    /// Experts in the model — every MoE layer's full complement.
    pub(crate) fn total_experts(&self) -> usize {
        self.num_moe_layers * self.experts_per_layer
    }

    /// Slots not currently holding an expert.
    pub(crate) fn free_len(&self) -> usize {
        self.zone.free_count()
    }

    /// Take the rightmost free slot, without evicting anything.
    ///
    /// `None` means every slot is occupied — the signal to consult the eviction
    /// policy, never a reason to skip it. Position decides *where*; temperature
    /// decides *who*.
    pub(crate) fn take_free(&mut self) -> Option<usize> {
        self.zone.alloc()
    }

    /// Return a slot whose contents are gone.
    pub(crate) fn put_free(&mut self, slot_idx: usize) {
        self.slots[slot_idx] = None;
        self.zone.release(slot_idx);
    }

    /// Device address of slot `slot_idx`'s first byte.
    pub(crate) fn slot_base(&self, slot_idx: usize) -> u64 {
        self.zone.slot_base(slot_idx)
    }

    /// Take `target - capacity` more slots from the KV side. Returns how many.
    ///
    /// The per-slot bookkeeping grows with the zone. Nothing is loaded into the
    /// new slots here — they join the free list and the next miss takes them,
    /// after every closer hole.
    pub(crate) fn grow_zone(&mut self, target: usize) -> usize {
        let gained = self.zone.grow_to(target);
        if gained > 0 {
            let n = self.zone.capacity();
            self.slots.resize_with(n, || None);
            self.last_used.resize(n, 0);
            self.slot_to_key.resize(n, None);
            self.publish_geometry();
        }
        gained
    }

    /// Republish the zone's shape after it has changed.
    ///
    /// Every mutation of `zone` must reach this, because the GPU dispatch
    /// tables' staleness check reads only what is published here — a zone that
    /// moves without publishing is one whose cached slot addresses go stale
    /// silently, which is the corruption this whole path exists to prevent.
    fn publish_geometry(&self) {
        let n = self.zone.capacity();
        self.geometry
            .publish(n, self.zone.slot_base(n.saturating_sub(1)));
    }

    /// Give `capacity - target` slots back to the KV side.
    ///
    /// Returns the plan the caller must perform on the bytes: relocate the
    /// hottest doomed occupants into surviving free slots, evict the rest. The
    /// bookkeeping *inside the zone* is already applied; the per-slot tables
    /// here are truncated once the caller has moved what it is going to move,
    /// which is why this returns before touching them.
    pub(crate) fn retract_zone(&mut self, target: usize) -> candle_nn::kv_cache::RetractPlan {
        // The pinned set does not move with the boundary — those experts have no
        // record in the pack and no warm slot, so unpinning one under pressure
        // would strand it. What keeps the eviction scan supplied instead is the
        // zone's floor: `minimum_resident_slots` prices the pinned set plus a
        // full working layer, and `WeightZone::retract_to` will not go below it.
        // (The failure this guards against is measured: a zone that retracted to
        // 297 slots against a 384-expert pinned set failed 1,774 consecutive
        // `allocate_slot` calls, because `layer >= pinned_layers` matched
        // nothing.)
        debug_assert!(
            target.max(self.zone.min_capacity())
                >= minimum_resident_slots(self.experts_per_layer).min(self.total_experts()),
            "retraction target is below the floor that prices the fixed pinned set"
        );
        // Keep key: pack-only experts survive a concession ahead of
        // warm-backed ones, then the hotter within each tier — the eviction
        // passes, inverted.
        let keep: Vec<(bool, f32)> = (0..self.zone.capacity())
            .map(|i| {
                self.slot_to_key[i].map_or((false, 0.0), |(layer, expert)| {
                    (
                        self.reload_tier(layer, expert) == ReloadTier::Nvme,
                        self.score(layer, expert),
                    )
                })
            })
            .collect();
        let before = self.zone.capacity();
        let plan = self.zone.retract_to(target, |i| keep[i]);
        // Ground has left the weight side. Recorded monotonically — see
        // `concede_epoch` for why comparing capacities later cannot stand in
        // for this, since the zone grows back.
        //
        // Keyed on the CAPACITY changing, not on the plan being non-empty: a
        // concession of slots that happened to be free moves the boundary and
        // hands their addresses to the tier just the same, while asking nothing
        // to be relocated or evicted. Those are precisely the slots a table
        // would keep pointing at with no other sign anything happened.
        if self.zone.capacity() != before {
            // Shape first, then the epoch: a reader that sees the bumped epoch
            // must not then read a pre-concession frontier and conclude the
            // zone is where it left it.
            self.publish_geometry();
            self.geometry.concede();
            // At INFO, not DEBUG. A concession retires expert slots and hands
            // their ground to the KV side; it is the event that turns an
            // all-resident grid into a paged one and invalidates every cached
            // slot address. It went unnoticed for a whole campaign because it
            // was logged at DEBUG while the daemon runs at INFO — 4.92 GiB of
            // expert ground changed hands and the log said nothing.
            tracing::info!(
                before,
                after = self.zone.capacity(),
                conceded = before.saturating_sub(self.zone.capacity()),
                frontier = format!("{:#x}", self.geometry.frontier()),
                concede_epoch = self.geometry.concede_epoch(),
                "expert cache: weight zone conceded ground to the KV side — cached slot \
                 addresses are now stale and GPU-native dispatch is refused"
            );
        }
        plan
    }

    /// Drop the per-slot tables to the zone's current capacity.
    ///
    /// Separate from [`Self::retract_zone`] because the caller has to move bytes
    /// and rewrite `slot_to_key` for the relocations in between; truncating
    /// first would take the entries it still needs to read.
    pub(crate) fn truncate_tables(&mut self) {
        let n = self.zone.capacity();
        self.slots.truncate(n);
        self.last_used.truncate(n);
        self.slot_to_key.truncate(n);
        // Anything the truncation removed is gone from VRAM; the location map
        // must not still point at it.
        self.key_to_slot.retain(|_, &mut slot| slot < n);
    }

    /// Promote a slot's timestamp (the hot path — one array write).
    #[inline]
    pub(crate) fn promote(&mut self, slot_idx: usize) {
        self.last_used[slot_idx] = self.generation;
        self.generation += 1;
    }

    /// Evict a slot: drop its contents and remove it from the lookup tables.
    ///
    /// Returns the evicted `(moe_layer, expert_idx)` key, or `None` if the slot
    /// was already empty.
    ///
    /// **Eviction moves no bytes.** The cold tier holds a valid copy of every
    /// expert at all times, so there is nothing to write back and nowhere to
    /// write it — this used to hand the `ExpertSlot` to the caller for a D2H
    /// copy into pinned RAM, and that copy duplicated data the pack file
    /// already held. Dropping the slot here releases the three `QMatMul` views;
    /// the zone owns the bytes and keeps them.
    pub(crate) fn evict(&mut self, slot_idx: usize) -> Option<(usize, usize)> {
        let evicted = self.slot_to_key[slot_idx];
        if let Some(evict_key) = evicted {
            self.key_to_slot.remove(&evict_key);
        }
        self.slot_to_key[slot_idx] = None;
        self.slots[slot_idx] = None;
        evicted
    }

    // ── Score update methods ─────────────────────────────────────────

    /// Index into `expert_scores` for a given (layer, expert) pair.
    #[inline]
    fn score_idx(&self, layer: usize, expert: usize) -> usize {
        layer * self.experts_per_layer + expert
    }

    /// Get the current score for a (layer, expert) pair.
    #[inline]
    pub(crate) fn score(&self, layer: usize, expert: usize) -> f32 {
        self.expert_scores[self.score_idx(layer, expert)]
    }

    /// Record a cache hit: bumps score by +1.0.
    #[inline]
    pub(crate) fn record_hit(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] += 1.0;
    }

    /// Record a successful speculative prediction: +0.3.
    #[inline]
    pub(crate) fn record_prediction_hit(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] += 0.3;
    }

    /// End-of-pass exponential decay: multiply all scores by `factor` (e.g. 0.85).
    pub(crate) fn decay_scores(&mut self, factor: f32) {
        for s in self.expert_scores.iter_mut() {
            *s *= factor;
        }
    }

    /// Forward (wrapped) distance from `current_layer` to `layer`: how many
    /// layers until the wave reaches `layer` again. Distance 0 = the layer being
    /// computed right now; distance `n-1` = the layer that just executed.
    #[inline]
    fn forward_distance(&self, layer: usize, current_layer: usize) -> usize {
        if layer >= current_layer {
            layer - current_layer
        } else {
            self.num_moe_layers - current_layer + layer
        }
    }

    /// Eviction score for a slot within its reload class
    /// ([`Self::cold_only`], which every scan compares first):
    /// `base_score × position_factor`. Lower = more likely to be evicted.
    ///
    /// `base_score` is the lightly-decayed access frequency — the dominant term,
    /// so frequently-reused experts stay resident (the cache is effectively LFU
    /// with a recency decay).  `position_factor` is a mild multiplier in
    /// `[0.5, 1.0]` that FALLS with forward (wrapped) reuse distance — Belady's
    /// direction: the layer about to be routed (distance 0) is most protected at
    /// 1.0, the just-executed layer (distance `n-1`, next use a full pass away)
    /// is the preferred victim near 0.5.
    #[inline]
    fn slot_eviction_score(&self, slot_idx: usize, current_layer: usize) -> f32 {
        if let Some(&(layer, expert)) = self.slot_to_key[slot_idx].as_ref() {
            let base = self.score(layer, expert);
            let n = self.num_moe_layers;
            let dist = self.forward_distance(layer, current_layer);
            let position_factor = 1.0 - 0.5 * (dist as f32 / n as f32);
            base * position_factor
        } else {
            0.0
        }
    }

    /// Allocate a free slot, evicting if necessary.
    ///
    /// ## Layer-aware eviction policy
    ///
    /// 1. **Free slots first** — no eviction cost.
    /// 2. **Behind-layer bias** — prefer evicting experts from layers that
    ///    have already executed in this pass (`PINNED_LAYERS <= layer < current_layer`).
    ///    Among those, pick the one with the lowest `slot_eviction_score`
    ///    (frequency × position factor), with recency as tie-breaker.
    /// 3. **Global score-based fallback** — if no behind-layer candidate exists
    ///    (e.g. early in the pass), fall back to the global lowest-score victim,
    ///    but still never evict pinned layers (0..PINNED_LAYERS-1).
    /// 4. **Pinned layers** — experts in layers 0..PINNED_LAYERS-1 are
    ///    never evicted.  They run first every pass with zero compute
    ///    overlap to hide DMA latency.
    ///
    /// Steps 2–3 run once per [`EVICTION_PASSES`] tier: over the warm-backed
    /// experts, and over the pack-only experts only when no warm-backed one is
    /// eligible.
    ///
    /// Returns `(slot_idx, evicted_key)`. `evicted_key` is `None` when a free
    /// slot was available and nothing was displaced.
    ///
    /// `protect` lists slots that must not be victims — the caller's hits and
    /// in-flight speculative/streamed installs. The latter matter here for
    /// more than waste: a streamed slot's bytes move on the STREAMER's
    /// stream, so re-tenanting it from this thread's copy stream is an
    /// unordered write race, not a benign overwrite.
    pub(crate) fn allocate_slot(
        &mut self,
        current_layer: usize,
        protect: &std::collections::HashSet<usize>,
    ) -> Result<(usize, Option<(usize, usize)>)> {
        // ── Try free slots first ──
        //
        // Drained before the policy is consulted at all: if any slot is free, no
        // eviction happens, whatever the scores say.
        if let Some(free) = self.zone.alloc() {
            return Ok((free, None));
        }

        for tier in EVICTION_PASSES {
            // ── Behind-layer scan: layers >= PINNED_LAYERS and < current_layer ──
            let behind = self.best_victim(current_layer, tier, |slot_idx, layer| {
                layer < current_layer && !protect.contains(&slot_idx)
            });
            // ── Global fallback (respects pinning + protection) ──
            let victim = behind.or_else(|| {
                self.best_victim(current_layer, tier, |slot_idx, _| {
                    !protect.contains(&slot_idx)
                })
            });
            if let Some(victim) = victim {
                return Ok((victim, self.evict(victim)));
            }
        }
        Err(candle::Error::Msg(
            "Expert cache full, cannot evict (all pinned)".into(),
        ))
    }

    /// The best single victim among occupied, non-pinned slots of reload
    /// `tier` that `eligible` admits (`(slot_idx, moe_layer)`): the lowest
    /// [`Self::slot_eviction_score`], then least recently used.
    fn best_victim(
        &self,
        current_layer: usize,
        tier: ReloadTier,
        eligible: impl Fn(usize, usize) -> bool,
    ) -> Option<usize> {
        self.slot_to_key
            .iter()
            .enumerate()
            .filter(|&(idx, k)| {
                k.is_some_and(|(layer, expert)| {
                    layer >= self.pinned_layers
                        && self.reload_tier(layer, expert) == tier
                        && eligible(idx, layer)
                })
            })
            .map(|(idx, _)| {
                (
                    idx,
                    self.slot_eviction_score(idx, current_layer),
                    self.last_used[idx],
                )
            })
            .min_by(|&(_, sa, la), &(_, sb, lb)| {
                sa.partial_cmp(&sb)
                    .unwrap_or(Ordering::Equal)
                    .then(la.cmp(&lb))
            })
            .map(|(idx, ..)| idx)
    }

    /// Up to `count` batch victims, ranked by [`VictimKey::order`] within each
    /// [`EVICTION_PASSES`] tier: the warm-backed pass first, and the pack-only
    /// pass only for what it could not supply. Best victim first.
    ///
    /// `weight(slot_idx, moe_layer, dist)` admits a non-pinned occupied slot
    /// by returning the factor its frequency is scaled by, or `None` to skip
    /// it.
    fn batch_victims(
        &self,
        current_layer: usize,
        count: usize,
        weight: impl Fn(usize, usize, usize) -> Option<f32>,
    ) -> Vec<usize> {
        let mut victims = Vec::with_capacity(count);
        for tier in EVICTION_PASSES {
            let need = count - victims.len();
            if need == 0 {
                break;
            }
            let mut cands: Vec<VictimKey> = self
                .slot_to_key
                .iter()
                .enumerate()
                .filter_map(|(idx, key)| {
                    let (layer, expert) = (*key)?;
                    if layer < self.pinned_layers || self.reload_tier(layer, expert) != tier {
                        return None;
                    }
                    let dist = self.forward_distance(layer, current_layer);
                    let factor = weight(idx, layer, dist)?;
                    Some(VictimKey {
                        slot: idx,
                        score: self.score(layer, expert) * factor,
                        dist,
                        lru: self.last_used[idx],
                    })
                })
                .collect();
            // O(n) partition to the best `need`, then order just those.
            if need < cands.len() {
                cands.select_nth_unstable_by(need, VictimKey::order);
                cands.truncate(need);
            }
            cands.sort_by(VictimKey::order);
            victims.extend(cands.iter().map(|c| c.slot));
        }
        victims
    }

    /// Free one slot to make room for a *prefetch*, choosing the safest victim
    /// among the **furthest** non-pinned layers.
    ///
    /// "Furthest" is relative with wraparound: forward distance is largest for
    /// the just-executed layer `current-1`, then `current-2`, … (the existing
    /// `slot_eviction_score` metric). Only the [`PREFETCH_EVICT_WINDOW`] furthest
    /// layers are eligible, so eviction never reaches the near-future layers
    /// about to be used — it stays within ~`current-1 .. current-PREFETCH_EVICT_WINDOW`
    /// (wrapping; at the pinned boundary that lands on the wave's tail).
    ///
    /// Within that window the choice is **frequency-dominated**: the
    /// least-used expert goes first (a never-used `L-3` is evicted before a hot
    /// `L-1`), then the farther one, then the LRU. Repeated calls therefore
    /// spread evictions across the window rather than draining one layer. The
    /// policy runs once per [`EVICTION_PASSES`] tier — warm-backed experts
    /// first, pack-only experts for whatever that pass could not free.
    ///
    /// Returns up to `count` `(slot_idx, evicted_key)` pairs (like
    /// [`Self::allocate_slot`]), fewer when the window is exhausted, empty if no
    /// eligible expert is resident.
    ///
    /// Batched on purpose: it scans the slot table **once** and partial-sorts the
    /// eligible candidates, rather than rescanning per victim. A dense prefill
    /// prefetch needs a whole layer's worth of slots, so the per-victim rescan
    /// would be O(slots × experts-per-layer) of pure CPU per layer.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn evict_for_prefetch_batch(
        &mut self,
        current_layer: usize,
        count: usize,
    ) -> Vec<(usize, Option<(usize, usize)>)> {
        if count == 0 {
            return Vec::new();
        }
        let min_dist = self.num_moe_layers.saturating_sub(PREFETCH_EVICT_WINDOW);
        // In-window only: anything nearer is about to be used.
        self.batch_victims(current_layer, count, |_, _, dist| {
            (dist >= min_dist).then_some(1.0)
        })
        .into_iter()
        .map(|slot| (slot, self.evict(slot)))
        .collect()
    }

    /// Batch-evict EXACTLY `count` victims (or as many as exist) so a layer's
    /// misses find free slots — the demand-sized replacement for the retired
    /// headroom guessing (per-layer drip + end-of-pass rate EMA). The caller is
    /// classify, which knows the exact deficit before any load, so eviction
    /// happens only on layers whose misses exceed the free list and never
    /// over-evicts.
    ///
    /// ## Victim key: `frequency × window_factor`, per reload tier
    ///
    /// Lowest key evicted first. The window factor is 0.5 for slots in the
    /// [`PREFETCH_EVICT_WINDOW`] layers directly behind the wave (wrapped
    /// forward distance `>= n - PREFETCH_EVICT_WINDOW` from `current_layer` —
    /// the just-executed layers, whose next use is a full pass away: Belady's
    /// choice) and 1.0 everywhere else, so the behind-window preference is
    /// worth a 2× frequency handicap.
    ///
    /// The policy runs once per [`EVICTION_PASSES`] tier: over the warm-backed
    /// experts first, and over the pack-only experts only for the shortfall.
    /// So the window never outranks the reload tier — a warm-backed expert
    /// ahead of the wave is evicted before a pack-only one just behind it. A
    /// HARD window tier was measured here and tripled cold pack reads
    /// (2.8k→8.2k at config-8, bulk −9%) precisely because it let window
    /// membership override the cold shield.
    ///
    /// Ties break farther-first then LRU, so repeated calls spread churn
    /// across the trailing layers instead of draining one.
    ///
    /// `protect` lists slot indices that must not be victims: the current
    /// layer's just-classified hits (about to be computed with) and in-flight
    /// prefetch installs for layers ahead of the wave (score ≈ 0 until their
    /// prediction validates — without protection they would be the
    /// coldest-looking slots on the card at exactly the moment they are most
    /// valuable).
    ///
    /// One O(slots) scan + an O(n) `select_nth` partition per tier pass.
    pub(crate) fn demand_eviction(
        &mut self,
        current_layer: usize,
        count: usize,
        protect: &[usize],
    ) -> Vec<(usize, usize)> {
        if count == 0 {
            return Vec::new();
        }
        let protected: std::collections::HashSet<usize> = protect.iter().copied().collect();
        let min_dist = self.num_moe_layers.saturating_sub(PREFETCH_EVICT_WINDOW);
        let victims = self.batch_victims(current_layer, count, |idx, _, dist| {
            if protected.contains(&idx) {
                None
            } else if dist >= min_dist {
                Some(0.5)
            } else {
                Some(1.0)
            }
        });

        let mut evicted_keys = Vec::with_capacity(victims.len());
        for slot_idx in victims {
            if let Some(key) = self.evict(slot_idx) {
                evicted_keys.push(key);
            }
            // Back to the zone, not to a local list: it decides where the next
            // load lands, and it is what the KV side reads to find the frontier.
            self.zone.release(slot_idx);
        }
        evicted_keys
    }

    /// Install an expert into a slot and update all lookup tables.
    pub(crate) fn install(
        &mut self,
        slot_idx: usize,
        moe_idx: usize,
        expert_idx: usize,
        slot: ExpertSlot,
    ) {
        self.slots[slot_idx] = Some(slot);
        self.key_to_slot.insert((moe_idx, expert_idx), slot_idx);
        self.slot_to_key[slot_idx] = Some((moe_idx, expert_idx));
        self.promote(slot_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache of `n` slots over a zone with no device behind it.
    ///
    /// The zone is pure arithmetic — addresses and indices — so the whole
    /// eviction policy still exercises with no GPU and no model load, exactly as
    /// it did when the free list was a local `Vec`.
    /// Pins the standard head layers, as every cache now does.
    fn cache(n: usize) -> ExpertCacheInner {
        ExpertCacheInner::new(WeightZone::new(1 << 30, 4096, n, n, 0), 48, 128)
    }

    /// A cache sized as the loader would size it.
    fn sized_cache(n: usize, experts_per_layer: usize) -> ExpertCacheInner {
        ExpertCacheInner::new(
            WeightZone::new(1 << 30, 4096, n, n, 0),
            48,
            experts_per_layer,
        )
    }

    /// **A cache at or above the floor can always evict**, at any size.
    ///
    /// The failure this rules out is not a slow cache, it is a dead one: once
    /// every resident slot holds a pinned-layer expert, the `layer >= pinned`
    /// filter matches nothing, and every load from then on fails — permanently,
    /// because escaping the state requires an eviction the state forbids. The
    /// daemon reached it with 297 slots against a 384-expert pinned set and
    /// failed 1,774 consecutive forwards.
    ///
    /// The pinned count is fixed, so the *floor* is what rules the state out:
    /// [`minimum_resident_slots`] prices the pinned set plus a full working
    /// layer. Swept over sizes from the floor upward, and over every model's
    /// expert count.
    #[test]
    fn a_cache_above_the_floor_always_has_a_victim() {
        for experts_per_layer in [8usize, 128, 256] {
            let floor = minimum_resident_slots(experts_per_layer);
            for capacity in [
                floor,
                floor + 1,
                4 * experts_per_layer,
                9 * experts_per_layer,
            ] {
                assert!(
                    PINNED_LAYERS * experts_per_layer + experts_per_layer <= capacity,
                    "the floor must leave the working layer room in {capacity} slots"
                );

                // The worst case that actually produced the deadlock: a batch
                // wide enough to fill every slot from layer 0 upward.
                let mut inner = sized_cache(capacity, experts_per_layer);
                for slot in 0..capacity {
                    let layer = slot / experts_per_layer;
                    occupy(&mut inner, slot, layer, slot % experts_per_layer, 0, 0.0);
                }
                assert!(
                    inner
                        .allocate_slot(capacity / experts_per_layer, &Default::default())
                        .is_ok(),
                    "a full cache of {capacity} slots at {experts_per_layer} experts \
                     a layer had no evictable slot"
                );
            }
        }
    }

    /// **The pinned depth never varies.** A pack omits the pinned layers'
    /// records, so a cache that unpinned a layer under pressure would strand it
    /// with nowhere to reload from — and a pack written on one machine would be
    /// unreadable on a smaller one. Capacity must not enter into it.
    #[test]
    fn the_pinned_depth_is_fixed_across_every_capacity() {
        for experts_per_layer in [8usize, 128, 256] {
            for capacity in [
                minimum_resident_slots(experts_per_layer),
                4 * experts_per_layer,
                9 * experts_per_layer,
                48 * experts_per_layer,
            ] {
                let inner = sized_cache(capacity, experts_per_layer);
                assert_eq!(
                    inner.pinned_layers, PINNED_LAYERS,
                    "{capacity} slots at {experts_per_layer}/layer changed the pinned depth"
                );
            }
        }
        // A model with fewer layers than the constant pins all of them.
        let tiny = ExpertCacheInner::new(WeightZone::new(1 << 30, 4096, 64, 64, 0), 1, 8);
        assert_eq!(tiny.pinned_layers, 1);
    }

    /// **A zone at the floor still has a victim**, which is the entire reason
    /// the floor exists.
    ///
    /// Fill every slot of a minimum-sized cache with pinned-layer experts — the
    /// worst case, a batch wide enough to route to all of them — and the scan
    /// must still find something to evict. Below this size it cannot: the
    /// pinned-layer filter matches nothing and every load from then on fails,
    /// permanently, because escaping the state requires an eviction the state
    /// forbids. The daemon sat at 297 slots against a 384-slot floor and failed
    /// 1,774 consecutive forwards that way.
    #[test]
    fn a_cache_at_its_floor_can_still_evict() {
        let experts_per_layer = 128;
        let floor = minimum_resident_slots(experts_per_layer);
        assert_eq!(floor, (PINNED_LAYERS + 1) * experts_per_layer + 1);

        let mut inner = cache(floor);
        // Every pinned-layer expert resident, and one slot beyond them.
        for slot in 0..floor {
            let layer = slot / experts_per_layer;
            occupy(&mut inner, slot, layer, slot % experts_per_layer, 0, 0.0);
        }
        assert!(
            inner
                .allocate_slot(PINNED_LAYERS, &Default::default())
                .is_ok(),
            "a zone sized to the pinned working set always has one slot the \
             scan is allowed to take"
        );

        // A cache with room for only the pinned set reaches the dead state: the
        // same fill leaves every resident slot holding a pinned-layer expert and
        // nothing can be freed. This is what the floor keeps the boundary out
        // of, with a full working layer of margin on top.
        let pinned_only = PINNED_LAYERS * experts_per_layer;
        assert!(pinned_only < floor, "the floor must clear the pinned set");
        let mut starved = cache(pinned_only);
        for slot in 0..pinned_only {
            let layer = slot / experts_per_layer;
            occupy(&mut starved, slot, layer, slot % experts_per_layer, 0, 0.0);
        }
        assert!(
            starved
                .allocate_slot(PINNED_LAYERS, &Default::default())
                .is_err(),
            "a cache holding only pinned-layer experts has nothing it may free"
        );
    }

    /// **The floor's arithmetic did not move when the pinned depth did.**
    ///
    /// This number feeds the opening boundary placement of every MoE model, and
    /// raising it starves the KV side (Qwen3-30B OOMs at load). Two pinned
    /// layers plus a working layer prices the same `3 × n + 1` that three
    /// capacity-derived layers did, so every model's placement is untouched.
    #[test]
    fn the_floor_is_arithmetically_unchanged() {
        for experts_per_layer in [8usize, 128, 256] {
            assert_eq!(
                minimum_resident_slots(experts_per_layer),
                3 * experts_per_layer + 1
            );
        }
        // The two models this actually places.
        assert_eq!(minimum_resident_slots(128), 385); // Qwen3-30B-A3B
        assert_eq!(minimum_resident_slots(256), 769); // Qwen3.5/3.6-35B-A3B
    }

    /// Mark a slot occupied by `(layer, expert)` without a real `ExpertSlot`
    /// (eviction selection reads only the bookkeeping tables, never the VRAM
    /// buffers).
    fn occupy(
        inner: &mut ExpertCacheInner,
        slot: usize,
        layer: usize,
        expert: usize,
        last_used: u32,
        freq: f32,
    ) {
        // The zone hands out the rightmost free index, and every test here
        // occupies ascending from 0, so the two agree. Asserting rather than
        // searching keeps the fixture honest: a test that stopped filling in
        // order would fail here rather than quietly occupy a different slot than
        // the one its assertions name.
        let taken = inner.zone.alloc().expect("a free slot");
        assert_eq!(taken, slot, "fixture must occupy slots in ascending order");
        inner.slot_to_key[slot] = Some((layer, expert));
        inner.key_to_slot.insert((layer, expert), slot);
        inner.last_used[slot] = last_used;
        inner.expert_scores[layer * inner.experts_per_layer + expert] = freq;
    }

    #[test]
    fn forced_eviction_targets_lowest_frequency() {
        // Four behind-the-wave experts at the same layer; the least-frequently
        // used (lowest score) is evicted, keeping the hot experts resident.
        let mut inner = cache(4);
        occupy(&mut inner, 0, 10, 100, 1, 8.0);
        occupy(&mut inner, 1, 10, 101, 2, 3.0);
        occupy(&mut inner, 2, 10, 102, 3, 0.5); // coldest
        occupy(&mut inner, 3, 10, 103, 4, 5.0);
        assert!(inner.free_len() == 0);

        let (slot, evicted_key) = inner.allocate_slot(20, &Default::default()).unwrap();
        assert_eq!(evicted_key, Some((10, 102)));
        assert_eq!(slot, 2);
    }

    #[test]
    fn demand_eviction_prefers_behind_window() {
        // current=35, n=48, window=5 → behind-window layers 30..34. The window
        // factor (0.5) is a 2× frequency handicap: a just-behind expert (layer
        // 30, dist 43) is evicted BEFORE a slightly-colder mid-distance one
        // (layer 20), steering demand churn onto the layers the wave just left.
        let mut inner = cache(4);
        occupy(&mut inner, 0, 1, 100, 1, 0.1); // pinned (layer < PINNED_LAYERS)
        occupy(&mut inner, 1, 10, 101, 2, 5.0);
        occupy(&mut inner, 2, 20, 102, 3, 0.4); // colder, but out of window (key 1.6·cost)
        occupy(&mut inner, 3, 30, 103, 4, 0.5); // in window (key 0.25·cost) → victim

        let evicted = inner.demand_eviction(35, 1, &[]);
        assert_eq!(evicted, vec![(30, 103)], "behind-window expert goes first");
        assert!(inner.key_to_slot.contains_key(&(10, 101)));
        assert!(inner.key_to_slot.contains_key(&(20, 102)));
        assert!(
            inner.key_to_slot.contains_key(&(1, 100)),
            "pinned layer was evicted"
        );
        assert_eq!(inner.free_len(), 1, "exactly the demanded count freed");
    }

    #[test]
    fn demand_eviction_frequency_ordered_within_window() {
        // Two behind-window candidates (layers 33 and 31 from current=35):
        // the least-used goes first, whatever its distance — the same
        // frequency-dominated order as the prefetch window.
        let mut inner = cache(3);
        occupy(&mut inner, 0, 33, 100, 1, 6.0); // in window, hot
        occupy(&mut inner, 1, 31, 101, 2, 0.3); // in window, coldest → victim
        occupy(&mut inner, 2, 34, 102, 3, 2.0); // in window, warm
        let evicted = inner.demand_eviction(35, 1, &[]);
        assert_eq!(evicted, vec![(31, 101)]);
    }

    #[test]
    fn demand_eviction_cold_shield_outranks_the_window() {
        // The 4× cold-reload penalty dominates the 2× window preference: a
        // warm-backed expert AHEAD of the wave (cheap RAM reload) is evicted
        // before an equally-used cold-only one just behind it (whose reload is
        // a pack read). The hard-tier variant inverted this trade and tripled
        // cold pack reads.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 32, 100, 1, 1.0); // in window, cold-only (key 4·0.5=2)
        occupy(&mut inner, 1, 40, 101, 2, 1.0); // ahead, warm-backed (key 1)
        inner.set_warm_backed(&[(40, 101)]);
        let evicted = inner.demand_eviction(35, 1, &[]);
        assert_eq!(
            evicted,
            vec![(40, 101)],
            "warm reload chosen over pack read"
        );
        assert!(inner.key_to_slot.contains_key(&(32, 100)));
    }

    #[test]
    fn demand_eviction_protects_the_layer_hits() {
        // A protected slot (the caller lists the current layer's hits) is
        // spared even when it is the coldest in-window candidate on the card;
        // the eviction takes the next-coldest window member instead.
        let mut inner = cache(3);
        occupy(&mut inner, 0, 30, 100, 1, 0.1); // in window, coldest — but a HIT, protected
        occupy(&mut inner, 1, 10, 101, 2, 5.0);
        occupy(&mut inner, 2, 33, 102, 3, 0.4); // in window, next-coldest → victim
        let evicted = inner.demand_eviction(35, 1, &[0]);
        assert_eq!(evicted, vec![(33, 102)], "protected hit slot spared");
        assert!(inner.key_to_slot.contains_key(&(30, 100)));
    }

    #[test]
    fn demand_eviction_protects_in_flight_installs() {
        // An in-flight prefetch install (near-future layer, score 0 — the
        // coldest-looking slot on the card) survives when listed in `protect`;
        // the eviction takes the next candidate instead.
        let mut inner = cache(3);
        occupy(&mut inner, 0, 36, 100, 1, 0.0); // in-flight install for L+1, protected
        occupy(&mut inner, 1, 20, 101, 2, 0.5); // out-of-window fallback → victim
        occupy(&mut inner, 2, 37, 102, 3, 4.0);
        let evicted = inner.demand_eviction(35, 1, &[0]);
        assert_eq!(evicted, vec![(20, 101)], "in-flight install spared");
        assert!(inner.key_to_slot.contains_key(&(36, 100)));
    }

    #[test]
    fn demand_eviction_caps_at_the_candidates() {
        // Asking for more than the non-pinned population frees what exists and
        // no more — the per-miss backstop in `allocate_slot` covers the rest.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 1, 100, 1, 0.1); // pinned
        occupy(&mut inner, 1, 10, 101, 2, 0.2);
        let evicted = inner.demand_eviction(20, 5, &[]);
        assert_eq!(evicted, vec![(10, 101)]);
        assert!(inner.key_to_slot.contains_key(&(1, 100)), "pinned survives");
    }

    #[test]
    fn allocate_slot_backstop_never_takes_a_protected_slot() {
        // The per-miss backstop must skip in-flight installs even when they
        // are the lowest-scored slots on the card: a streamed slot's bytes
        // move on another stream, so re-tenanting it is a write race.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 36, 100, 1, 0.0); // in-flight stream install, protected
        occupy(&mut inner, 1, 40, 101, 2, 9.0); // hot, but the only legal victim
        let protect: std::collections::HashSet<usize> = [0].into_iter().collect();
        let (slot, evicted_key) = inner.allocate_slot(35, &protect).unwrap();
        assert_eq!(evicted_key, Some((40, 101)), "protected slot skipped");
        assert_eq!(slot, 1);
        assert!(inner.key_to_slot.contains_key(&(36, 100)));
    }

    #[test]
    fn allocate_slot_global_fallback_prefers_furthest_future() {
        // No behind-layer candidates (everything resident is ahead of the
        // wave), equal frequency: the corrected position factor evicts the
        // FURTHEST-future expert (next use latest — Belady), not the one about
        // to be routed. The inverted factor chose (36, 100) here.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 36, 100, 1, 2.0); // L+1 — about to be routed, kept
        occupy(&mut inner, 1, 45, 101, 2, 2.0); // L+10 — furthest future → victim
        let (slot, evicted_key) = inner.allocate_slot(35, &Default::default()).unwrap();
        assert_eq!(evicted_key, Some((45, 101)));
        assert_eq!(slot, 1);
    }

    #[test]
    fn prefetch_evict_is_frequency_dominated_in_window() {
        // current=10, n=48, window=5 → eligible layers 5..9 (the 5 just-behind).
        // A never-used expert at L-3 (layer 7) is evicted before a hot expert at
        // the furthest L-1 (layer 9): usage dominates distance.
        let mut inner = cache(4);
        occupy(&mut inner, 0, 9, 100, 5, 8.0); // L-1, furthest, but hot
        occupy(&mut inner, 1, 7, 102, 5, 0.0); // L-3, never used
        occupy(&mut inner, 2, 30, 103, 5, 9.0); // out of window (dist 20)
        let (slot, key) = inner
            .evict_for_prefetch_batch(10, 1)
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(key, Some((7, 102)), "never-used L-3 evicted over hot L-1");
        assert_eq!(slot, 1);
    }

    #[test]
    fn prefetch_evict_prefers_farther_among_equally_cold() {
        // Two never-used experts in-window → the farther (L-1) goes first.
        let mut inner = cache(4);
        occupy(&mut inner, 0, 9, 100, 5, 0.0); // L-1 (dist 47), cold
        occupy(&mut inner, 1, 6, 101, 5, 0.0); // L-4 (dist 44), cold
        let (_, key) = inner
            .evict_for_prefetch_batch(10, 1)
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(key, Some((9, 100)), "farther of two cold experts evicted");
    }

    #[test]
    fn prefetch_evict_protects_near_future_even_if_unused() {
        // current=10, window=5: a never-used near-future expert (layer 12, dist 2)
        // is OUT of window and must be protected; only the in-window (hot) expert
        // is eligible.
        let mut inner = cache(4);
        occupy(&mut inner, 0, 12, 200, 5, 0.0); // near-future, never used — protected
        occupy(&mut inner, 1, 8, 201, 5, 9.0); // L-2, in window, hot
        let (_, key) = inner
            .evict_for_prefetch_batch(10, 1)
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(
            key,
            Some((8, 201)),
            "near-future layer never evicted for prefetch"
        );
    }

    #[test]
    fn prefetch_evict_at_pinned_boundary_lands_on_tail() {
        // current=2, n=62, window=5: the window (L-1..L-5 = layers 1,0,61,60,59)
        // has only the tail layers 59..61 non-pinned. A never-used near-future
        // layer (5, dist 3) is out of window and protected.
        let mut inner = ExpertCacheInner::new(WeightZone::new(1 << 30, 4096, 4, 4, 0), 62, 128);
        occupy(&mut inner, 0, 61, 100, 5, 1.0); // tail, in window
        occupy(&mut inner, 1, 5, 101, 5, 0.0); // near-future (dist 3), protected
        let (_, key) = inner
            .evict_for_prefetch_batch(2, 1)
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(key, Some((61, 100)), "pinned boundary evicts the tail");
    }

    #[test]
    fn prefetch_evict_batch_returns_victims_in_priority_order() {
        // One scan yields multiple victims, best-first: equally-cold L-1 then L-2;
        // the hot L-3 is left resident. Exercises the dense double-buffer path.
        let mut inner = cache(4);
        occupy(&mut inner, 0, 9, 100, 5, 0.0); // L-1, cold
        occupy(&mut inner, 1, 8, 101, 5, 0.0); // L-2, cold
        occupy(&mut inner, 2, 7, 102, 5, 5.0); // L-3, hot — kept
        let victims = inner.evict_for_prefetch_batch(10, 2);
        assert_eq!(victims.len(), 2);
        assert_eq!(victims[0].1, Some((9, 100)));
        assert_eq!(victims[1].1, Some((8, 101)));
        assert!(inner.key_to_slot.contains_key(&(7, 102)), "hot expert kept");
    }

    /// A cache holding nothing but pinned-layer experts offers the prefetcher no
    /// victims — sized off [`PINNED_LAYERS`] so it stays the "all pinned" case
    /// if the depth ever changes.
    #[test]
    fn prefetch_evict_none_when_all_pinned() {
        let mut inner = cache(PINNED_LAYERS);
        for layer in 0..PINNED_LAYERS {
            occupy(&mut inner, layer, layer, 100 + layer, layer as u32 + 1, 0.0);
        }
        assert!(inner.evict_for_prefetch_batch(5, 1).is_empty());
    }

    #[test]
    fn pinned_layers_never_evicted() {
        let mut inner = cache(PINNED_LAYERS);
        for layer in 0..PINNED_LAYERS {
            occupy(&mut inner, layer, layer, 100 + layer, layer as u32 + 1, 0.0);
        }
        // Every resident expert is pinned → no legal victim → error.
        assert!(inner.allocate_slot(5, &Default::default()).is_err());
    }

    /// An expert with no warm copy costs an NVMe read to bring back, so it is
    /// kept in preference to an equally-cold one that reloads over PCIe.
    #[test]
    fn the_expert_with_no_warm_copy_is_kept() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 5, 1.0); // warm-backed
        occupy(&mut inner, 1, 10, 51, 5, 1.0); // cold-only, same temperature
        inner.set_warm_backed(&[(10, 50)]);

        let (_, evicted_key) = inner.allocate_slot(20, &Default::default()).unwrap();
        assert_eq!(
            evicted_key,
            Some((10, 50)),
            "evicted the expert that would have to come back from disk"
        );
    }

    /// The reload tier is a pass, not a tilt: even a hot warm-backed expert
    /// goes before a pack-only one nobody is using.
    #[test]
    fn a_hot_warm_backed_expert_goes_before_a_cold_pack_only_one() {
        let mut inner = cache(5);
        occupy(&mut inner, 0, 10, 50, 1, 40.0); // warm-backed and very hot
        occupy(&mut inner, 1, 11, 51, 2, 0.1); // pack-only and cold
        occupy(&mut inner, 2, 12, 52, 3, 0.2);
        occupy(&mut inner, 3, 13, 53, 4, 0.3);
        occupy(&mut inner, 4, 14, 54, 5, 0.4);
        inner.set_warm_backed(&[(10, 50)]);

        let (slot, evicted_key) = inner.allocate_slot(20, &Default::default()).unwrap();
        assert_eq!(evicted_key, Some((10, 50)));
        assert_eq!(slot, 0);
    }

    /// The single-victim path runs its whole policy — behind-layer scan, then
    /// the global fallback — over the warm-backed experts before it considers
    /// a pack-only one, so a warm-backed expert AHEAD of the wave goes before a
    /// pack-only one behind it.
    #[test]
    fn allocate_slot_exhausts_the_warm_pass_before_the_pack() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 50, 1, 0.0); // behind, pack-only, never used
        occupy(&mut inner, 1, 30, 51, 2, 7.0); // ahead, warm-backed, hot
        occupy(&mut inner, 2, 40, 52, 3, 2.0); // ahead, warm-backed, cooler
        inner.set_warm_backed(&[(30, 51), (40, 52)]);

        let (slot, evicted_key) = inner.allocate_slot(20, &Default::default()).unwrap();
        assert_eq!(evicted_key, Some((40, 52)), "coolest warm-backed expert");
        assert_eq!(slot, 2);
        // Within the warm pass the policy still prefers behind the wave.
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 50, 1, 9.0); // behind, warm-backed, hot
        occupy(&mut inner, 1, 30, 51, 2, 0.0); // ahead, warm-backed, cold
        occupy(&mut inner, 2, 12, 52, 3, 0.0); // behind, pack-only, cold
        inner.set_warm_backed(&[(10, 50), (30, 51)]);
        let (_, evicted_key) = inner.allocate_slot(20, &Default::default()).unwrap();
        assert_eq!(
            evicted_key,
            Some((10, 50)),
            "behind-layer bias inside the pass"
        );
    }

    /// The batch path takes what the warm pass can supply, then runs the same
    /// policy over the pack-only experts for the shortfall — each pass in its
    /// own frequency × window order.
    #[test]
    fn demand_eviction_runs_the_pack_pass_only_for_the_shortfall() {
        let mut inner = cache(5);
        occupy(&mut inner, 0, 10, 50, 1, 6.0); // warm
        occupy(&mut inner, 1, 11, 51, 2, 3.0); // warm
        occupy(&mut inner, 2, 12, 52, 3, 0.5); // pack-only, coldest on the card
        occupy(&mut inner, 3, 13, 53, 4, 2.0); // pack-only
        occupy(&mut inner, 4, 32, 54, 5, 3.0); // pack-only, in window: key 1.5
        inner.set_warm_backed(&[(10, 50), (11, 51)]);

        let evicted = inner.demand_eviction(35, 4, &[]);
        assert_eq!(
            evicted,
            vec![(11, 51), (10, 50), (12, 52), (32, 54)],
            "both warm (by frequency), then the two best pack-only (0.5, then 1.5 over 2.0)"
        );
        assert!(inner.key_to_slot.contains_key(&(13, 53)));
        assert_eq!(inner.free_len(), 4);
    }

    /// With no warm tier every expert is pack-only, and the passes collapse to
    /// the plain policy: the same victims in the same order.
    #[test]
    fn with_no_warm_tier_the_policy_is_unchanged() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 50, 1, 6.0);
        occupy(&mut inner, 1, 20, 51, 2, 0.5);
        occupy(&mut inner, 2, 33, 52, 3, 0.8); // in window: key 0.4
        let evicted = inner.demand_eviction(35, 2, &[]);
        assert_eq!(evicted, vec![(33, 52), (20, 51)]);
    }

    /// A concession inverts the passes: a warm-backed expert is dropped before
    /// a pack-only one, even a colder one.
    #[test]
    fn a_concession_relocates_the_pack_only_expert() {
        let mut inner = sized_cache(6, 1);
        for slot in 0..6 {
            occupy(&mut inner, slot, 10 + slot, 0, slot as u32, 1.0);
        }
        // One free destination below the new frontier.
        inner.evict(0);
        inner.put_free(0);
        inner.expert_scores[15] = 0.5; // slot 5: pack-only, colder
        inner.expert_scores[14] = 9.0; // slot 4: warm-backed and hot
        inner.set_warm_backed(&[(14, 0)]);

        let plan = inner.retract_zone(4);
        assert_eq!(plan.relocate, vec![(5, 0)]);
        assert_eq!(plan.evict, vec![4]);
    }

    /// The prefetch make-room path weighs it too — it is the same choice.
    #[test]
    fn prefetch_eviction_also_prefers_the_warm_backed_victim() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 9, 100, 5, 1.0); // L-1, warm-backed
        occupy(&mut inner, 1, 9, 101, 5, 1.0); // L-1, cold-only
        inner.set_warm_backed(&[(9, 100)]);
        let (_, key) = inner
            .evict_for_prefetch_batch(10, 1)
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(key, Some((9, 100)));
    }

    #[test]
    fn hot_expert_survives_a_cold_one() {
        // Same layer (same position factor): the frequently-used expert is kept
        // and the cold one evicted — the cache is frequency-dominated.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 9, 9.0);
        occupy(&mut inner, 1, 10, 51, 1, 0.0);

        let (_, evicted_key) = inner.allocate_slot(20, &Default::default()).unwrap();
        assert_eq!(
            evicted_key,
            Some((10, 51)),
            "cold expert was not evicted in preference to the hot one"
        );
    }

    #[test]
    fn behind_layer_preferred_over_ahead() {
        // A (hot) expert behind the wave is still evicted in preference to a
        // (cold) one ahead of it — never drop an expert not yet executed this
        // pass, so eviction can never cascade into later layers.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 5, 5.0); // behind, hot
        occupy(&mut inner, 1, 30, 51, 0, 0.0); // ahead, cold

        let (_, evicted_key) = inner.allocate_slot(20, &Default::default()).unwrap();
        assert_eq!(
            evicted_key,
            Some((10, 50)),
            "evicted an expert that is still ahead of the wave"
        );
    }
}
