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
//! 1. **End-of-pass batch eviction** — after the last MoE layer, evict the
//!    lowest-scored occupied slots to create free headroom for the next pass.
//! 2. **Layer-aware forced eviction** — on a miss with no free slot, prefer
//!    evicting a low-scored expert from a layer already executed this pass
//!    (behind the wave, so it can never cascade), then fall back to the global
//!    lowest-scored victim.
//! 3. **Early-layer pinning** — the first [`PINNED_LAYERS`] layers are never
//!    evicted (they run first every pass with no compute to hide a reload).
//! 4. **Free-slot-only prefetch** — speculative prefetch never evicts
//!    (enforced by the pipeline, not this module).
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
use std::collections::HashMap;

/// Number of early MoE layers whose experts are never evicted.
///
/// These layers run first every pass and have zero compute to overlap
/// with DMA — evicting them guarantees cold misses with maximum stall.
/// Pinning layers 0–2 locks in ~24 experts (top-8 × 3 layers), a
/// negligible fraction of the 2,805 slot budget.
pub(crate) const PINNED_LAYERS: usize = 3;

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
/// Each slot is either free (in `free_slots`), or occupied (has an
/// `ExpertSlot` and a `slot_to_key` entry).  Occupied slots have a
/// `last_used` timestamp that determines eviction order.
///
/// ```text
/// Free:     slots[i] = None,  slot_to_key[i] = None
/// Occupied: slots[i] = Some(ExpertSlot), slot_to_key[i] = Some((moe, exp))
/// ```
pub struct ExpertCacheInner {
    /// VRAM slots — created on-demand, indexed by slot_idx.
    /// **No Arc wrapping** — sole ownership.
    pub(crate) slots: Vec<Option<ExpertSlot>>,
    /// Free slot indices (populated at init, drained on first loads).
    pub(crate) free_slots: Vec<usize>,
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
}

impl ExpertCacheInner {
    /// Create a new empty cache with `num_slots` free slots.
    ///
    /// * `num_moe_layers` — total MoE layers (e.g. 48)
    /// * `experts_per_layer` — experts per layer (e.g. 128)
    pub(crate) fn new(num_slots: usize, num_moe_layers: usize, experts_per_layer: usize) -> Self {
        Self {
            slots: (0..num_slots).map(|_| None).collect(),
            free_slots: (0..num_slots).rev().collect(),
            key_to_slot: HashMap::new(),
            last_used: vec![0u32; num_slots],
            generation: 0,
            slot_to_key: vec![None; num_slots],
            expert_scores: vec![0.0f32; num_moe_layers * experts_per_layer],
            num_moe_layers,
            experts_per_layer,
        }
    }

    /// Promote a slot's timestamp (the hot path — one array write).
    #[inline]
    pub(crate) fn promote(&mut self, slot_idx: usize) {
        self.last_used[slot_idx] = self.generation;
        self.generation += 1;
    }

    /// Evict a slot: remove from lookup tables, return the VRAM buffers.
    ///
    /// Returns the evicted `(moe_layer, expert_idx)` key **and** the
    /// `ExpertSlot` so the caller can D2H copy data to the pinned pool
    /// before dropping the VRAM buffers.
    pub(crate) fn evict(
        &mut self,
        slot_idx: usize,
    ) -> (Option<(usize, usize)>, Option<ExpertSlot>) {
        let evicted = self.slot_to_key[slot_idx];
        if let Some(evict_key) = evicted {
            self.key_to_slot.remove(&evict_key);
        }
        self.slot_to_key[slot_idx] = None;
        let slot = self.slots[slot_idx].take();
        (evicted, slot)
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

    /// Combined eviction score for a slot: `base_score × position_factor`.
    /// Lower = more likely to be evicted.
    ///
    /// `base_score` is the lightly-decayed access frequency — the dominant term,
    /// so frequently-reused experts stay resident (the cache is effectively LFU
    /// with a recency decay).  `position_factor` is a mild multiplier in
    /// `[0.5, 1.0]` that rises with reuse distance, gently preferring to evict
    /// experts whose next use is sooner among equally-cold candidates.
    #[inline]
    fn slot_eviction_score(&self, slot_idx: usize, current_layer: usize) -> f32 {
        if let Some(&(layer, expert)) = self.slot_to_key[slot_idx].as_ref() {
            let base = self.score(layer, expert);
            let n = self.num_moe_layers;
            let forward_distance = if layer >= current_layer {
                layer - current_layer
            } else {
                n - current_layer + layer
            };
            let position_factor = 0.5 + 0.5 * (forward_distance as f32 / n as f32);
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
    /// Returns `(slot_idx, evicted_key, evicted_slot)`.
    /// `evicted_key` is `None` when a free slot was available (no eviction).
    /// `evicted_slot` is returned so the caller can D2H copy to pinned RAM.
    pub(crate) fn allocate_slot(
        &mut self,
        current_layer: usize,
    ) -> Result<(usize, Option<(usize, usize)>, Option<ExpertSlot>)> {
        // ── Try free slots first ──
        if let Some(free) = self.free_slots.pop() {
            return Ok((free, None, None));
        }

        // ── Behind-layer scan: layers >= PINNED_LAYERS and < current_layer ──
        // Pick the slot with the lowest eviction score among behind-layer experts.
        let mut behind_slot: Option<usize> = None;
        let mut behind_score: f32 = f32::MAX;
        let mut behind_lru: u32 = u32::MAX;

        for (slot_idx, key) in self.slot_to_key.iter().enumerate() {
            if let Some((moe_layer, _)) = key {
                if *moe_layer < PINNED_LAYERS || *moe_layer >= current_layer {
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
            let (evicted_key, evicted_slot) = self.evict(victim);
            return Ok((victim, evicted_key, evicted_slot));
        }

        // ── Global score-based fallback (respects pinning) ──
        // Pick the slot with the lowest eviction score globally.
        let victim = self
            .slot_to_key
            .iter()
            .enumerate()
            .filter(|(_, k)| k.map_or(false, |(layer, _)| layer >= PINNED_LAYERS))
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
        let (evicted_key, evicted_slot) = self.evict(victim);
        Ok((victim, evicted_key, evicted_slot))
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
    /// Returns up to `count` `(slot_idx, evicted_key, evicted_slot)` tuples (like
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
    ) -> Vec<(usize, Option<(usize, usize)>, Option<ExpertSlot>)> {
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
                    let dist = if layer >= current_layer {
                        layer - current_layer
                    } else {
                        n - current_layer + layer
                    };
                    if dist < min_dist {
                        return None; // too near — protect the upcoming layers
                    }
                    Some((idx, self.score(layer, expert), dist, self.last_used[idx]))
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
            .map(|(idx, _, _, _)| {
                let (key, slot) = self.evict(idx);
                (idx, key, slot)
            })
            .collect()
    }

    /// Evict the bottom `fraction` of occupied slots by eviction score.
    ///
    /// Called at the end of each forward pass (after the last MoE layer)
    /// to create free headroom for the next pass.  The freed slots become
    /// available for both real misses and speculative prefetch without
    /// any eviction during the pass.
    ///
    /// Uses `slot_eviction_score` (frequency × position factor) at
    /// `current_layer = 0` (start of next pass) so that behind-layer experts
    /// from the end of the previous pass get lower priority.
    ///
    /// Respects pinning: experts in layers 0..PINNED_LAYERS-1 are never
    /// evicted.
    ///
    /// Returns evicted `((moe_layer, expert_idx), ExpertSlot)` pairs so
    /// the caller can D2H copy them to the pinned pool before dropping.
    pub(crate) fn end_of_pass_eviction(
        &mut self,
        fraction: f32,
    ) -> Vec<((usize, usize), ExpertSlot)> {
        // Collect (slot_idx, eviction_score) for all non-pinned occupied slots.
        // Use current_layer = 0 since we're at end of pass / about to start
        // a new pass.
        let mut candidates: Vec<(usize, f32)> = self
            .slot_to_key
            .iter()
            .enumerate()
            .filter_map(|(idx, key)| {
                key.and_then(|(layer, _)| {
                    if layer >= PINNED_LAYERS {
                        Some((idx, self.slot_eviction_score(idx, 0)))
                    } else {
                        None
                    }
                })
            })
            .collect();

        if candidates.is_empty() {
            return Vec::new();
        }

        let evict_count = ((candidates.len() as f32 * fraction).ceil() as usize)
            .max(1)
            .min(candidates.len());

        // O(n) partial sort: partition so that candidates[..evict_count]
        // contains the lowest-scored elements (in arbitrary order).
        if evict_count < candidates.len() {
            candidates.select_nth_unstable_by(evict_count, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let mut evicted_pairs = Vec::with_capacity(evict_count);
        for &(slot_idx, _) in candidates[..evict_count].iter() {
            let (evicted_key, evicted_slot) = self.evict(slot_idx);
            if let (Some(key), Some(slot)) = (evicted_key, evicted_slot) {
                evicted_pairs.push((key, slot));
            }
            self.free_slots.push(slot_idx);
        }
        evicted_pairs
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

    /// Mark a slot occupied by `(layer, expert)` without a real `ExpertSlot`
    /// (eviction selection reads only the bookkeeping tables, never the VRAM
    /// buffers), so the policy can be exercised with no GPU or model load.
    fn occupy(
        inner: &mut ExpertCacheInner,
        slot: usize,
        layer: usize,
        expert: usize,
        last_used: u32,
        freq: f32,
    ) {
        inner.free_slots.retain(|&s| s != slot);
        inner.slot_to_key[slot] = Some((layer, expert));
        inner.key_to_slot.insert((layer, expert), slot);
        inner.last_used[slot] = last_used;
        inner.expert_scores[layer * inner.experts_per_layer + expert] = freq;
    }

    #[test]
    fn forced_eviction_targets_lowest_frequency() {
        // Four behind-the-wave experts at the same layer; the least-frequently
        // used (lowest score) is evicted, keeping the hot experts resident.
        let mut inner = ExpertCacheInner::new(4, 48, 128);
        occupy(&mut inner, 0, 10, 100, 1, 8.0);
        occupy(&mut inner, 1, 10, 101, 2, 3.0);
        occupy(&mut inner, 2, 10, 102, 3, 0.5); // coldest
        occupy(&mut inner, 3, 10, 103, 4, 5.0);
        assert!(inner.free_slots.is_empty());

        let (slot, evicted_key, _) = inner.allocate_slot(20).unwrap();
        assert_eq!(evicted_key, Some((10, 102)));
        assert_eq!(slot, 2);
    }

    #[test]
    fn end_of_pass_evicts_lowest_scored() {
        // End-of-pass drops the lowest-scored non-pinned slot and keeps the hot
        // ones; pinned layers are never considered.
        let mut inner = ExpertCacheInner::new(4, 48, 128);
        occupy(&mut inner, 0, 1, 100, 1, 0.1); // pinned (layer < PINNED_LAYERS)
        occupy(&mut inner, 1, 10, 101, 2, 5.0);
        occupy(&mut inner, 2, 20, 102, 3, 0.2); // coldest non-pinned
        occupy(&mut inner, 3, 30, 103, 4, 4.0);

        let _ = inner.end_of_pass_eviction(0.1); // ceil(3 × 0.1) = 1 of 3 non-pinned
        assert!(
            !inner.key_to_slot.contains_key(&(20, 102)),
            "coldest non-pinned expert not evicted"
        );
        assert!(inner.key_to_slot.contains_key(&(10, 101)));
        assert!(inner.key_to_slot.contains_key(&(30, 103)));
        assert!(
            inner.key_to_slot.contains_key(&(1, 100)),
            "pinned layer was evicted"
        );
    }

    #[test]
    fn prefetch_evict_is_frequency_dominated_in_window() {
        // current=10, n=48, window=5 → eligible layers 5..9 (the 5 just-behind).
        // A never-used expert at L-3 (layer 7) is evicted before a hot expert at
        // the furthest L-1 (layer 9): usage dominates distance.
        let mut inner = ExpertCacheInner::new(4, 48, 128);
        occupy(&mut inner, 0, 9, 100, 5, 8.0); // L-1, furthest, but hot
        occupy(&mut inner, 1, 7, 102, 5, 0.0); // L-3, never used
        occupy(&mut inner, 2, 30, 103, 5, 9.0); // out of window (dist 20)
        let (slot, key, _) = inner
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
        let mut inner = ExpertCacheInner::new(4, 48, 128);
        occupy(&mut inner, 0, 9, 100, 5, 0.0); // L-1 (dist 47), cold
        occupy(&mut inner, 1, 6, 101, 5, 0.0); // L-4 (dist 44), cold
        let (_, key, _) = inner
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
        let mut inner = ExpertCacheInner::new(4, 48, 128);
        occupy(&mut inner, 0, 12, 200, 5, 0.0); // near-future, never used — protected
        occupy(&mut inner, 1, 8, 201, 5, 9.0); // L-2, in window, hot
        let (_, key, _) = inner
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
        let mut inner = ExpertCacheInner::new(4, 62, 128);
        occupy(&mut inner, 0, 61, 100, 5, 1.0); // tail, in window
        occupy(&mut inner, 1, 5, 101, 5, 0.0); // near-future (dist 3), protected
        let (_, key, _) = inner
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
        let mut inner = ExpertCacheInner::new(4, 48, 128);
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
        let mut inner = ExpertCacheInner::new(3, 48, 128);
        occupy(&mut inner, 0, 0, 100, 1, 0.0);
        occupy(&mut inner, 1, 1, 101, 2, 0.0);
        occupy(&mut inner, 2, 2, 102, 3, 0.0);
        assert!(inner.evict_for_prefetch_batch(5, 1).is_empty());
    }

    #[test]
    fn pinned_layers_never_evicted() {
        let mut inner = ExpertCacheInner::new(3, 48, 128);
        occupy(&mut inner, 0, 0, 100, 1, 0.0);
        occupy(&mut inner, 1, 1, 101, 2, 0.0);
        occupy(&mut inner, 2, 2, 102, 3, 0.0);
        // Every resident expert is pinned → no legal victim → error.
        assert!(inner.allocate_slot(5).is_err());
    }

    #[test]
    fn hot_expert_survives_a_cold_one() {
        // Same layer (same position factor): the frequently-used expert is kept
        // and the cold one evicted — the cache is frequency-dominated.
        let mut inner = ExpertCacheInner::new(2, 48, 128);
        occupy(&mut inner, 0, 10, 50, 9, 9.0);
        occupy(&mut inner, 1, 10, 51, 1, 0.0);

        let (_, evicted_key, _) = inner.allocate_slot(20).unwrap();
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
        let mut inner = ExpertCacheInner::new(2, 48, 128);
        occupy(&mut inner, 0, 10, 50, 5, 5.0); // behind, hot
        occupy(&mut inner, 1, 30, 51, 0, 0.0); // ahead, cold

        let (_, evicted_key, _) = inner.allocate_slot(20).unwrap();
        assert_eq!(
            evicted_key,
            Some((10, 50)),
            "evicted an expert that is still ahead of the wave"
        );
    }
}
