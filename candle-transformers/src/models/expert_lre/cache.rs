//! Expert cache bookkeeping — slot management, eviction policy, score-based.
//!
//! This module contains [`ExpertCacheInner`], the mutable bookkeeping
//! structure that tracks which experts are resident in VRAM, manages
//! slot allocation, and implements the score-based eviction policy.
//!
//! ## Eviction policy
//!
//! See the module-level docs in [`super`] for the full policy description.
//! In brief:
//!
//! 1. **End-of-pass batch eviction** — after the last MoE layer, evict
//!    the lowest-scored occupied slots.
//! 2. **Layer-aware forced eviction** — prefer evicting behind-layer
//!    experts with low scores, then fall back to global score-based.
//! 3. **Early-layer pinning** — first [`PINNED_LAYERS`] layers are
//!    never evicted.
//! 4. **Free-slot-only prefetch** — speculative prefetch never evicts
//!    (enforced by the pipeline, not this module).
//!
//! ## Score table
//!
//! A flat `Vec<f32>` indexed by `layer * experts_per_layer + expert`.
//! Updated by pipeline events:
//!
//! - **Cache hit**: +1.0
//! - **Prediction hit**: +0.3 (speculative load that was actually needed)
//! - **Anti-prediction**: −0.2 (bottom-N from transition matrix, clamped ≥ 0)
//! - **End-of-pass decay**: ×0.85 (exponential forgetting)

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
    /// Higher score = more valuable = evicted last.
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

    /// Anti-prediction penalty: −0.2, clamped at 0.0.
    #[inline]
    pub(crate) fn record_anti_prediction(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] = (self.expert_scores[idx] - 0.2).max(0.0);
    }

    /// End-of-pass exponential decay: multiply all scores by `factor` (e.g. 0.85).
    pub(crate) fn decay_scores(&mut self, factor: f32) {
        for s in self.expert_scores.iter_mut() {
            *s *= factor;
        }
    }

    /// Combined eviction score for a slot: `base_score × position_factor`.
    /// Lower = more likely to be evicted.
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
    ///    (score × position_factor), with recency as tie-breaker.
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

    /// Evict the bottom `fraction` of occupied slots by eviction score.
    ///
    /// Called at the end of each forward pass (after the last MoE layer)
    /// to create free headroom for the next pass.  The freed slots become
    /// available for both real misses and speculative prefetch without
    /// any eviction during the pass.
    ///
    /// Uses `slot_eviction_score` (base_score × position_factor) at
    /// `current_layer = 0` (start of next pass) so that behind-layer
    /// experts from the end of the previous pass get lower priority.
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
