//! The slot and score tables that decide which expert holds which VRAM slot and
//! which one the next miss evicts.
//!
//! The cache owns no bytes. The weight zone owns the slot addresses and is the
//! free list, and an eviction only drops the tables and the views, because the
//! cold pack holds a valid copy of every expert and nothing needs writing back.
//! A victim is ranked by its frequency scaled by how far behind the wave its
//! layer lies, but the reload tier and whether its row routed it last time run
//! as ordered passes ahead of that score, so the cheapest expected miss is paid
//! first. Free slots are taken rightmost, which keeps live experts packed away
//! from the boundary the KV side pushes against.

use super::*;

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
            decode_reuse: vec![0.0f32; num_moe_layers * experts_per_layer],
            num_moe_layers,
            experts_per_layer,
            pinned_layers: pinned_layer_count(num_moe_layers),
            warm_backed: Arc::new(HostBacked::new(num_moe_layers, experts_per_layer)),
            hit_last: vec![false; num_moe_layers * experts_per_layer],
            passes: EVICTION_PASSES,
            #[cfg(feature = "tensor-assert")]
            owners: None,
        }
    }

    /// Tag every slot with its tenant, and write through every tenant a slot
    /// gains from here on — once the startup fill has installed its experts.
    #[cfg(feature = "tensor-assert")]
    pub(crate) fn attach_owners(&mut self, owners: Arc<SlotOwners>) {
        for (slot, key) in self.slot_to_key.iter().enumerate() {
            if let Some((row, expert)) = *key {
                owners.set(slot, row, expert);
            }
        }
        self.owners = Some(owners);
    }

    /// Slot `slot` gained the tenant `slot_to_key` names: write its tag.
    #[cfg(feature = "tensor-assert")]
    pub(crate) fn mirror_tenant(&self, slot: usize) {
        if let (Some(owners), Some((row, expert))) = (&self.owners, self.slot_to_key[slot]) {
            owners.set(slot, row, expert);
        }
    }

    /// Record which experts the warm tier holds — each a resident pinned slot
    /// or a pageable copy — once the startup fill is decided; the stager keeps
    /// it current from then on.
    pub(crate) fn set_warm_backed(&mut self, membership: &[(usize, usize)]) {
        self.warm_backed.clear();
        for &(layer, expert) in membership {
            self.warm_backed.set(layer, expert, true);
        }
    }

    /// Row `layer` was served, routing `routed`: those of its experts are now
    /// the ones it hit on its last visit, and no others.
    pub(crate) fn mark_visit(&mut self, layer: usize, routed: &[usize]) {
        let row = layer * self.experts_per_layer;
        self.hit_last[row..row + self.experts_per_layer].fill(false);
        for &e in routed {
            self.hit_last[row + e] = true;
        }
    }

    /// Order the eviction passes by each class's expected miss cost, cheapest
    /// first: `cost[hit * 2 + warm]` is how often a victim of that class is
    /// routed again at its row's next visit times what restoring it costs.
    /// Ties keep [`EVICTION_PASSES`]'s order, which takes the warm-backed
    /// first.
    ///
    /// Restoring a pack-only expert costs the drive read *and* the copy that
    /// follows, against the copy alone for a warm-backed one, so pack-only
    /// classes are protected by that ratio — about an order of magnitude on an
    /// NVMe pack — and an eviction takes one only when it is that much less
    /// likely to be wanted back.
    pub(crate) fn set_eviction_costs(&mut self, cost: [f64; 4]) {
        let mut passes = EVICTION_PASSES;
        passes.sort_by(|&a, &b| cost[pass_class(a)].total_cmp(&cost[pass_class(b)]));
        self.passes = passes;
    }

    /// Whether `layer`'s row routed `expert` on its last visit.
    #[inline]
    pub(crate) fn hit_last_visit(&self, layer: usize, expert: usize) -> bool {
        self.hit_last[layer * self.experts_per_layer + expert]
    }

    /// Whether a warm (host) copy backs `(layer, expert)` — its cheap reload.
    #[inline]
    pub(crate) fn is_warm_backed(&self, layer: usize, expert: usize) -> bool {
        self.reload_tier(layer, expert) == ReloadTier::Warm
    }

    /// Where `(layer, expert)` reloads from after an eviction — see
    /// [`ReloadTier`] for why eviction passes split by tier.
    #[inline]
    pub(super) fn reload_tier(&self, layer: usize, expert: usize) -> ReloadTier {
        if self.warm_backed.get(layer, expert) {
            ReloadTier::Warm
        } else {
            ReloadTier::Nvme
        }
    }

    /// Apply a protection factor to a base score, so that "larger factor ⇒
    /// better protected" holds on both sides of zero.
    ///
    /// [`behind_protection`] means "how much this slot is worth protecting", so
    /// a larger one must always raise the key. A plain multiply only does that
    /// above zero. [`PREFILL_ELEVATE_PENALTY`] drives a freshly prefill-loaded
    /// expert *below* zero on purpose, and multiplying a negative base by a
    /// protection factor makes it more negative: the slot about to run would be
    /// taken before the one just behind — exactly backwards, and invisible,
    /// because the result is a plausible number in the wrong order.
    ///
    /// Dividing below zero fixes the direction while keeping the sign a hard
    /// tier: whatever its protection, an elevated expert still sorts below every
    /// scored one in its pass, which is the separation
    /// [`PREFILL_ELEVATE_PENALTY`] exists to create. `protect` is always in
    /// `[FADE_FLOOR, 1]`, so this never divides by zero.
    #[inline]
    pub(super) fn protected_by(base: f32, protect: f32) -> f32 {
        if base < 0.0 {
            base / protect
        } else {
            base * protect
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

    /// Slots the pinned layers fill — experts with no copy in the pack or the
    /// warm tier, which no retraction may drop ([`ZoneFloor::Paused`]'s floor).
    pub(crate) fn pinned_slots(&self) -> usize {
        self.pinned_layers * self.experts_per_layer
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
    pub(crate) fn retract_zone(
        &mut self,
        target: usize,
        within: ZoneFloor,
    ) -> candle_nn::kv_cache::RetractPlan {
        // The pinned set does not move with the boundary — those experts have no
        // record in the pack and no warm slot, so unpinning one under pressure
        // would strand it. What keeps the eviction scan supplied instead is the
        // zone's floor: `minimum_resident_slots` prices the pinned set plus a
        // full working layer, and `WeightZone::retract_to` will not go below it.
        // (The failure this guards against is measured: a zone that retracted to
        // 297 slots against a 384-expert pinned set failed 1,774 consecutive
        // slot allocations, because `layer >= pinned_layers` matched nothing.)
        //
        // **Paused, the working layer goes too** (`ZoneFloor::Paused`): nothing
        // forwards for the length of a guest drain, so nothing needs the room to
        // serve a token, and the drain grows the zone back to its floor before
        // the engine forwards again. The pinned set stays either way.
        let floor = match within {
            ZoneFloor::Serving => self.zone.min_capacity(),
            ZoneFloor::Paused => self.pinned_slots().min(self.zone.capacity()),
        };
        debug_assert!(
            target.max(floor) >= self.pinned_slots().min(self.zone.capacity()),
            "retraction target is below the fixed pinned set"
        );
        // Keep key: a pinned-layer expert above everything — it has nowhere to
        // reload from, and a zone retracted to the pinned set holds exactly it —
        // then pack-only experts ahead of warm-backed ones, then the hotter
        // within each tier: the eviction passes, inverted. The same key decides
        // which survivor below the frontier a hotter doomed expert displaces; a
        // pinned-layer expert is never displaced.
        let pinned_layers = self.pinned_layers;
        let keep: Vec<(bool, bool, f32)> = (0..self.zone.capacity())
            .map(|i| {
                self.slot_to_key[i].map_or((false, false, 0.0), |(layer, expert)| {
                    (
                        layer < pinned_layers,
                        self.reload_tier(layer, expert) == ReloadTier::Nvme,
                        self.score(layer, expert),
                    )
                })
            })
            .collect();
        let before = self.zone.capacity();
        let slot_to_key = &self.slot_to_key;
        let plan = self.zone.retract_within(
            target,
            floor,
            |i| keep[i],
            |i| slot_to_key[i].is_some_and(|(layer, _)| layer >= pinned_layers),
        );
        // Keyed on the CAPACITY changing, not on the plan being non-empty: a
        // concession of slots that happened to be free moves the boundary just
        // the same, while asking nothing to be relocated or evicted.
        if self.zone.capacity() != before {
            // At INFO, not DEBUG. A concession retires expert slots and hands
            // their ground to the KV side; it is the event that turns an
            // all-resident grid into a paged one. It went unnoticed for a whole
            // campaign because it was logged at DEBUG while the daemon runs at
            // INFO — 4.92 GiB of expert ground changed hands and the log said
            // nothing.
            let n = self.zone.capacity();
            tracing::info!(
                before,
                after = n,
                conceded = before.saturating_sub(n),
                frontier = format!("{:#x}", self.zone.slot_base(n.saturating_sub(1))),
                "expert cache: weight zone conceded ground to the KV side"
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
    pub(super) fn score_idx(&self, layer: usize, expert: usize) -> usize {
        layer * self.experts_per_layer + expert
    }

    /// Get the current score for a (layer, expert) pair.
    #[inline]
    pub(crate) fn score(&self, layer: usize, expert: usize) -> f32 {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] + DECODE_REUSE_WEIGHT * self.decode_reuse[idx]
    }

    /// Record a decode hit: +1.0, and +1.0 of long-memory decode reuse.
    #[inline]
    pub(crate) fn record_hit(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] += 1.0;
        self.decode_reuse[idx] += 1.0;
    }

    /// Record a decode miss: +1.0, so the expert holds its slot to the next
    /// step, which near-certainly routes to it again — but no decode reuse
    /// until it is hit.
    #[inline]
    pub(crate) fn record_decode_miss(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] += 1.0;
    }

    /// Record a prefill-attributed hit on an already-resident expert:
    /// [`PREFILL_HIT_SCORE`] (+0.1) rather than the full decode weight.
    #[inline]
    pub(crate) fn record_prefill_hit(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] += PREFILL_HIT_SCORE;
    }

    /// Record a prefill-attributed elevation (a fresh cold load, not a hit):
    /// [`PREFILL_ELEVATE_PENALTY`] (−0.1), biasing it toward the next
    /// eviction scan rather than leaving its score at whatever an earlier,
    /// unrelated occupancy left behind.
    ///
    /// **Assigned, not accumulated.** `install` does not clear `expert_scores`,
    /// so the slot's previous tenant's score is still standing when a fresh
    /// expert lands on it. Adding the penalty to that leaves a formerly
    /// decode-hot expert at, say, 11.9 — comfortably protected, which is the
    /// opposite of what an elevation means and exactly the "earlier, unrelated
    /// occupancy" the constant's doc says this must not inherit. An elevation is
    /// one event with one meaning: this expert was just paid for and is the
    /// least likely thing in the zone to be wanted again — by recency.
    ///
    /// **Decode reuse is kept.** It belongs to the expert, not to the slot, and
    /// was earned by decode hits on this same expert, so an expert decode keeps
    /// returning to still outranks a prompt's one-shot fetch after a prompt
    /// happens to load it: its score is −0.1 plus [`DECODE_REUSE_WEIGHT`] × its
    /// reuse.
    #[inline]
    pub(crate) fn record_prefill_elevate(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] = PREFILL_ELEVATE_PENALTY;
    }

    /// Record a successful speculative prediction: +0.3.
    #[inline]
    pub(crate) fn record_prediction_hit(&mut self, layer: usize, expert: usize) {
        let idx = self.score_idx(layer, expert);
        self.expert_scores[idx] += 0.3;
    }

    /// End-of-pass exponential decay: multiply all scores by `factor` (e.g. 0.85),
    /// and decode reuse by [`DECODE_REUSE_DECAY`].
    pub(crate) fn decay_scores(&mut self, factor: f32) {
        for s in self.expert_scores.iter_mut() {
            *s *= factor;
        }
        for r in self.decode_reuse.iter_mut() {
            *r *= DECODE_REUSE_DECAY;
        }
    }

    /// Forward (wrapped) distance from `current_layer` to `layer`: how many
    /// layers until the wave reaches `layer` again. Distance 0 = the layer being
    /// computed right now; distance `n-1` = the layer that just executed.
    #[inline]
    pub(super) fn forward_distance(&self, layer: usize, current_layer: usize) -> usize {
        if layer >= current_layer {
            layer - current_layer
        } else {
            self.num_moe_layers - current_layer + layer
        }
    }

    /// Up to `count` batch victims, ranked by [`VictimKey::order`] within each
    /// [`EVICTION_PASSES`] pass: a later pass only for what the earlier ones
    /// could not supply. Best victim first.
    ///
    /// `weight(slot_idx, moe_layer, dist)` admits a non-pinned occupied slot
    /// by returning the protection factor its frequency is scaled by (through
    /// [`Self::protected_by`], so an elevated expert's negative score is not
    /// inverted), or `None` to skip it.
    pub(super) fn batch_victims(
        &self,
        current_layer: usize,
        count: usize,
        weight: impl Fn(usize, usize, usize) -> Option<f32>,
    ) -> Vec<usize> {
        let mut victims = Vec::with_capacity(count);
        for (tier, hit) in self.passes {
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
                    if layer < self.pinned_layers
                        || self.reload_tier(layer, expert) != tier
                        || self.hit_last_visit(layer, expert) != hit
                    {
                        return None;
                    }
                    let dist = self.forward_distance(layer, current_layer);
                    let factor = weight(idx, layer, dist)?;
                    Some(VictimKey {
                        slot: idx,
                        score: Self::protected_by(self.score(layer, expert), factor),
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

    /// Up to `count` victims, best first, **without evicting them** — the
    /// caller decides, per victim, whether it may be taken now (the reclaim
    /// rule: an expert whose row is still being computed cannot give up its
    /// slot) and evicts the ones it takes with [`Self::evict`].
    ///
    /// ## Victim key: `frequency × behind_protection`, per reload tier
    ///
    /// Lowest key first. The protection fades with how far behind the wave a
    /// layer is ([`behind_protection`]): L−1 is scaled by [`FADE_FLOOR`] and
    /// each layer further back a little more, wrapping round to the layers about
    /// to run, which keep nearly their whole score. So the wave protects what
    /// is ahead of it at the expense of what is behind, and an expert just
    /// behind survives one about to run only with about `1 / FADE_FLOOR` its
    /// score — enough to steer churn behind the wave, not enough to evict a hot
    /// expert the next pass needs for a cold one ahead.
    ///
    /// Demand and read-ahead claims spend the same ranked offers, so a read-ahead
    /// copy for a row ahead is bought from the layers the wave has left, not
    /// from the ones it is about to route.
    ///
    /// The policy runs once per eviction pass — a reload tier × whether the
    /// row routed the expert on its last visit — cheapest expected miss cost
    /// first ([`Self::set_eviction_costs`]), and a later pass only for the
    /// shortfall. So neither distance nor frequency outranks a pass: a
    /// warm-backed expert ahead of the wave goes before a pack-only one just
    /// behind it, and a stale hot expert before one its row just routed,
    /// whenever the measured costs order the passes that way. A hard
    /// distance tier across the reload tiers was measured here and tripled cold
    /// pack reads (2.8k→8.2k at config-8, bulk −9%) precisely because it let
    /// position override the cold shield.
    ///
    /// Ties break farther-first then LRU, so a never-used expert goes from the
    /// layer just behind first. Pinned layers are never candidates.
    /// `admit(slot, layer)` refuses a slot outright.
    ///
    /// One O(slots) scan + an O(n) `select_nth` partition per tier pass.
    pub(crate) fn rank_victims(
        &self,
        current_layer: usize,
        count: usize,
        admit: impl Fn(usize, usize) -> bool,
    ) -> Vec<usize> {
        if count == 0 {
            return Vec::new();
        }
        let n = self.num_moe_layers;
        self.batch_victims(current_layer, count, |idx, layer, dist| {
            admit(idx, layer).then(|| behind_protection(dist, n))
        })
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
        #[cfg(feature = "tensor-assert")]
        self.mirror_tenant(slot_idx);
        self.promote(slot_idx);
    }
}
