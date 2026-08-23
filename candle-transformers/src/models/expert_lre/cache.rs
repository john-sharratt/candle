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
use candle::Result;
use candle_nn::kv_cache::WeightZone;
use std::collections::HashMap;

/// Number of early MoE layers whose experts are never evicted.
///
/// These layers run first every pass and have zero compute to overlap
/// with DMA — evicting them guarantees cold misses with maximum stall.
/// A single decode step routes top-8 per layer, so pinning layers 0–2 holds
/// ~24 experts; a batch wide enough to route everywhere holds all of them, which
/// is what [`minimum_resident_slots`] prices.
pub(crate) const PINNED_LAYERS: usize = 3;

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
/// The daemon reached it: the boundary retracted to 297 slots against
/// 3 × 128 = 384 pinned-eligible, and the next 1,774 wave steps all failed
/// identically. This is the number that must never be crossed, and since the
/// boundary is otherwise free to trade expert residency for KV ground on demand,
/// it is the **only** thing standing between a hungry KV side and a dead engine.
///
/// One slot on top of the pinned set, so the load that triggered an eviction has
/// somewhere to land.
pub fn minimum_resident_slots(experts_per_layer: usize) -> usize {
    PINNED_LAYERS * experts_per_layer + 1
}

/// How much more an expert with no warm copy is worth keeping, per unit of
/// temperature.
///
/// The ratio of what the two reloads cost is about 8× — ~1 ms for a 2.9 MB
/// positioned NVMe read against ~116 µs for the same bytes H2D from pinned host
/// memory at PCIe bandwidth — but the value here is **4, and that is measured,
/// not derived**. At 8 the term stops being a tilt on the ordering and starts
/// replacing it: cold-only experts are held past the point where their
/// temperature justifies it, the cache's hit rate falls (44.8 % → 44.3 % on
/// Q8_0×20), and the 43 further cold reads it saves cost more than they buy.
/// Every config was slower at 8 than at 4; the widest lost 55 t/s.
///
/// At 4, frequency still decides among experts of equal reload cost and a truly
/// cold cold-backed expert still loses to a hot warm-backed one. What it stops
/// is the policy evicting the expensive one when the two are otherwise close —
/// which, at the margin an eviction scan actually operates on, is most of them.
const COLD_RELOAD_PENALTY: f32 = 4.0;

/// How many of the furthest (just-behind, wrapping) layers are eligible as
/// prefetch make-room victims. Caps how far back eviction reaches from the
/// current layer (`current-1 .. current-PREFETCH_EVICT_WINDOW`), keeping it off
/// the near-future layers about to be used. At the pinned boundary this window
/// lands on the wave's tail. See [`ExpertCacheInner::evict_for_prefetch_batch`].
#[cfg(any(feature = "cuda", test))]
pub(crate) const PREFETCH_EVICT_WINDOW: usize = 5;

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
        Self {
            slots: (0..num_slots).map(|_| None).collect(),
            zone,
            key_to_slot: HashMap::new(),
            last_used: vec![0u32; num_slots],
            generation: 0,
            slot_to_key: vec![None; num_slots],
            expert_scores: vec![0.0f32; num_moe_layers * experts_per_layer],
            num_moe_layers,
            experts_per_layer,
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

    /// What it costs to bring `(layer, expert)` back after evicting it, relative
    /// to the cheapest case.
    ///
    /// **An eviction policy that ignores this is choosing blind.** Under the
    /// three-tier cache the two outcomes differ by an order of magnitude: an
    /// expert the warm tier holds comes back as a ~116 µs H2D from pinned host
    /// memory, and one it does not comes back as a 2.9 MB positioned NVMe read
    /// on the pipeline thread — page-cache-bypassing, so a real device round
    /// trip, measured near a millisecond. The old two-tier cache had no such
    /// distinction to make: every expert not in VRAM was in pinned RAM by
    /// construction, so every reload cost the same and the score could be pure
    /// temperature.
    ///
    /// Weighting the score by it makes the cache converge on the right shape
    /// without anyone choosing it: VRAM drifts toward holding the experts that
    /// are expensive to re-acquire, the warm tier covers the ones that are
    /// cheap, and the experts that churn are the ones whose churn is cheapest.
    #[inline]
    fn reload_cost(&self, layer: usize, expert: usize) -> f32 {
        let idx = layer * self.experts_per_layer + expert;
        if self.warm_backed.get(idx).copied().unwrap_or(false) {
            1.0
        } else {
            COLD_RELOAD_PENALTY
        }
    }

    /// Slots that exist — the zone's capacity, which is also `slots.len()`.
    pub(crate) fn num_slots(&self) -> usize {
        self.zone.capacity()
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
        }
        gained
    }

    /// Give `capacity - target` slots back to the KV side.
    ///
    /// Returns the plan the caller must perform on the bytes: relocate the
    /// hottest doomed occupants into surviving free slots, evict the rest. The
    /// bookkeeping *inside the zone* is already applied; the per-slot tables
    /// here are truncated once the caller has moved what it is going to move,
    /// which is why this returns before touching them.
    pub(crate) fn retract_zone(&mut self, target: usize) -> candle_nn::kv_cache::RetractPlan {
        let scores: Vec<f32> = (0..self.zone.capacity())
            .map(|i| self.slot_to_key[i].map_or(0.0, |(layer, expert)| self.score(layer, expert)))
            .collect();
        self.zone.retract_to(target, |i| scores[i])
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

    /// Combined eviction score for a slot:
    /// `base_score × position_factor × reload_cost`. Lower = more likely to be
    /// evicted.
    ///
    /// `base_score` is the lightly-decayed access frequency — the dominant term,
    /// so frequently-reused experts stay resident (the cache is effectively LFU
    /// with a recency decay).  `position_factor` is a mild multiplier in
    /// `[0.5, 1.0]` that FALLS with forward (wrapped) reuse distance — Belady's
    /// direction: the layer about to be routed (distance 0) is most protected at
    /// 1.0, the just-executed layer (distance `n-1`, next use a full pass away)
    /// is the preferred victim near 0.5.
    /// [`Self::reload_cost`] is what it would take to undo the eviction, which
    /// is the term that keeps an expert with no warm copy out of the disk path.
    #[inline]
    fn slot_eviction_score(&self, slot_idx: usize, current_layer: usize) -> f32 {
        if let Some(&(layer, expert)) = self.slot_to_key[slot_idx].as_ref() {
            let base = self.score(layer, expert);
            let n = self.num_moe_layers;
            let dist = self.forward_distance(layer, current_layer);
            let position_factor = 1.0 - 0.5 * (dist as f32 / n as f32);
            base * position_factor * self.reload_cost(layer, expert)
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

        // ── Behind-layer scan: layers >= PINNED_LAYERS and < current_layer ──
        // Pick the slot with the lowest eviction score among behind-layer experts.
        let mut behind_slot: Option<usize> = None;
        let mut behind_score: f32 = f32::MAX;
        let mut behind_lru: u32 = u32::MAX;

        for (slot_idx, key) in self.slot_to_key.iter().enumerate() {
            if let Some((moe_layer, _)) = key {
                if *moe_layer < PINNED_LAYERS
                    || *moe_layer >= current_layer
                    || protect.contains(&slot_idx)
                {
                    continue;
                }
                let es = self.slot_eviction_score(slot_idx, current_layer);
                let lru = self.last_used[slot_idx];
                if es < behind_score || (es == behind_score && lru < behind_lru) {
                    behind_slot = Some(slot_idx);
                    behind_score = es;
                    behind_lru = lru;
                }
            }
        }

        if let Some(victim) = behind_slot {
            return Ok((victim, self.evict(victim)));
        }

        // ── Global score-based fallback (respects pinning + protection) ──
        // Pick the slot with the lowest eviction score globally.
        let victim = self
            .slot_to_key
            .iter()
            .enumerate()
            .filter(|(idx, k)| {
                k.is_some_and(|(layer, _)| layer >= PINNED_LAYERS) && !protect.contains(idx)
            })
            .min_by(|(idx_a, _), (idx_b, _)| {
                let sa = self.slot_eviction_score(*idx_a, current_layer);
                let sb = self.slot_eviction_score(*idx_b, current_layer);
                sa.partial_cmp(&sb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| self.last_used[*idx_a].cmp(&self.last_used[*idx_b]))
            })
            .map(|(idx, _)| idx)
            .ok_or_else(|| {
                candle::Error::Msg("Expert cache full, cannot evict (all pinned)".into())
            })?;
        Ok((victim, self.evict(victim)))
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
    /// spread evictions across the window rather than draining one layer.
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
        let n = self.num_moe_layers;
        let min_dist = n.saturating_sub(PREFETCH_EVICT_WINDOW);
        // One scan: collect eligible (in-window, non-pinned) candidates with
        // their sort keys.
        let mut cands: Vec<(usize, f32, usize, u32)> = self
            .slot_to_key
            .iter()
            .enumerate()
            .filter_map(|(idx, key)| {
                key.and_then(|(layer, expert)| {
                    if layer < PINNED_LAYERS {
                        return None;
                    }
                    let dist = self.forward_distance(layer, current_layer);
                    if dist < min_dist {
                        return None; // too near — protect the upcoming layers
                    }
                    // Weighted like the demand path: a never-used expert with no
                    // warm copy is still worth more than a never-used one that
                    // reloads over PCIe.
                    Some((
                        idx,
                        self.score(layer, expert) * self.reload_cost(layer, expert),
                        dist,
                        self.last_used[idx],
                    ))
                })
            })
            .collect();
        // Best victims first: least-used, then farthest, then LRU.
        cands.sort_by(|&(_, sa, da, la), &(_, sb, db, lb)| {
            sa.partial_cmp(&sb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(db.cmp(&da))
                .then(la.cmp(&lb))
        });
        cands.truncate(count);
        cands
            .into_iter()
            .map(|(idx, _, _, _)| (idx, self.evict(idx)))
            .collect()
    }

    /// Batch-evict EXACTLY `count` victims (or as many as exist) so a layer's
    /// misses find free slots — the demand-sized replacement for the retired
    /// headroom guessing (per-layer drip + end-of-pass rate EMA). The caller is
    /// classify, which knows the exact deficit before any load, so eviction
    /// happens only on layers whose misses exceed the free list and never
    /// over-evicts.
    ///
    /// ## Victim key: `frequency × reload_cost × window_factor`
    ///
    /// Lowest key evicted first. The window factor is 0.5 for slots in the
    /// [`PREFETCH_EVICT_WINDOW`] layers directly behind the wave (wrapped
    /// forward distance `>= n - PREFETCH_EVICT_WINDOW` from `current_layer` —
    /// the just-executed layers, whose next use is a full pass away: Belady's
    /// choice) and 1.0 everywhere else. That makes the behind-window
    /// preference worth a 2× frequency handicap while [`Self::reload_cost`]'s
    /// cold penalty ([`COLD_RELOAD_PENALTY`] = 4×) stays dominant: a
    /// warm-backed expert ahead of the wave is still evicted before a
    /// cold-only one just behind it. A HARD window tier was measured here and
    /// tripled cold pack reads (2.8k→8.2k at config-8, bulk −9%) precisely
    /// because it let window membership override the cold shield.
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
    /// One O(slots) scan + an O(n) `select_nth` partition per call, the same
    /// amortization the batch paths always had.
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
        let n = self.num_moe_layers;
        let min_dist = n.saturating_sub(PREFETCH_EVICT_WINDOW);
        // (slot, freq × reload_cost × window_factor, dist, lru)
        let mut candidates: Vec<(usize, f32, usize, u32)> = self
            .slot_to_key
            .iter()
            .enumerate()
            .filter_map(|(idx, key)| {
                key.and_then(|(layer, expert)| {
                    if layer < PINNED_LAYERS || protected.contains(&idx) {
                        return None;
                    }
                    let dist = self.forward_distance(layer, current_layer);
                    let window_factor = if dist >= min_dist { 0.5 } else { 1.0 };
                    Some((
                        idx,
                        self.score(layer, expert) * self.reload_cost(layer, expert) * window_factor,
                        dist,
                        self.last_used[idx],
                    ))
                })
            })
            .collect();

        if candidates.is_empty() {
            return Vec::new();
        }

        let evict_count = count.min(candidates.len());

        // O(n) partial sort: partition so that candidates[..evict_count]
        // contains the best victims (lowest key; farther then LRU on ties).
        if evict_count < candidates.len() {
            candidates.select_nth_unstable_by(evict_count, |&(_, sa, da, la), &(_, sb, db, lb)| {
                sa.partial_cmp(&sb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(db.cmp(&da))
                    .then(la.cmp(&lb))
            });
        }

        let mut evicted_keys = Vec::with_capacity(evict_count);
        for &(slot_idx, ..) in candidates[..evict_count].iter() {
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
    fn cache(n: usize) -> ExpertCacheInner {
        ExpertCacheInner::new(WeightZone::new(1 << 30, 4096, n, n, 0), 48, 128)
    }

    /// **A zone at the floor still has a victim**, which is the entire reason
    /// the floor exists.
    ///
    /// Fill every slot of a minimum-sized cache with pinned-layer experts — the
    /// worst case, a batch wide enough to route to all of them — and the scan
    /// must still find something to evict. Below this size it cannot: the
    /// `layer >= PINNED_LAYERS` filter matches nothing and every load from then
    /// on fails, permanently, because escaping the state requires an eviction
    /// the state forbids. The daemon sat at 297 slots against a 384-expert
    /// pinned set and failed 1,774 consecutive forwards that way.
    #[test]
    fn a_cache_at_its_floor_can_still_evict() {
        let experts_per_layer = 128;
        let floor = minimum_resident_slots(experts_per_layer);
        assert_eq!(floor, PINNED_LAYERS * experts_per_layer + 1);

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

        // One slot short, the same fill leaves nothing evictable — the state
        // the floor exists to keep the boundary out of.
        let mut starved = cache(floor - 1);
        for slot in 0..floor - 1 {
            let layer = slot / experts_per_layer;
            occupy(&mut starved, slot, layer, slot % experts_per_layer, 0, 0.0);
        }
        assert!(
            starved
                .allocate_slot(PINNED_LAYERS, &Default::default())
                .is_err(),
            "below the floor every resident slot is pinned and nothing can be freed"
        );
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

    #[test]
    fn prefetch_evict_none_when_all_pinned() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 0, 100, 1, 0.0);
        occupy(&mut inner, 1, 1, 101, 2, 0.0);
        occupy(&mut inner, 2, 2, 102, 3, 0.0);
        assert!(inner.evict_for_prefetch_batch(5, 1).is_empty());
    }

    #[test]
    fn pinned_layers_never_evicted() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 0, 100, 1, 0.0);
        occupy(&mut inner, 1, 1, 101, 2, 0.0);
        occupy(&mut inner, 2, 2, 102, 3, 0.0);
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

    /// The penalty is a tilt, not an override: a genuinely hot warm-backed
    /// expert still outranks a cold-only one nobody is using.
    #[test]
    fn temperature_still_outranks_the_reload_cost() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 5, 40.0); // warm-backed but very hot
        occupy(&mut inner, 1, 10, 51, 5, 1.0); // cold-only and cold
        inner.set_warm_backed(&[(10, 50)]);

        let (_, evicted_key) = inner.allocate_slot(20, &Default::default()).unwrap();
        assert_eq!(
            evicted_key,
            Some((10, 51)),
            "the reload penalty overrode a 40x temperature difference"
        );
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
