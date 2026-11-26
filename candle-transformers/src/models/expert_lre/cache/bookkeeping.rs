//! The slot and score tables that decide which expert holds which VRAM slot and
//! which one the next miss evicts.
//!
//! The cache owns no bytes. The weight zone owns the slot addresses and is the
//! free list, and an eviction only drops the tables and the views, because the
//! cold pack holds a valid copy of every expert and nothing needs writing back.
//! A victim is ranked by its frequency scaled by how far behind the wave its
//! row lies and by a small protection for a pack-only reload, all in one key;
//! only whether its row routed it last time runs as a pass ahead of that key.
//! Free slots are taken rightmost, which keeps live experts packed away from the
//! boundary the KV side pushes against.
//!
//! Every expert is keyed `(model, row, expert)` ([`ExpertKey`]). Distance behind
//! the wave is defined within the model that is forwarding: a row of any other
//! model is maximally behind, so only its hits keep it (`docs/parasite_model.md`
//! §11.3).

use super::*;

impl ExpertCacheInner {
    /// Create a new empty cache over `zone`'s slots, for the models `models`
    /// registers.
    pub(crate) fn new(zone: WeightZone, models: CardModels) -> Self {
        let num_slots = zone.capacity();
        let total = models.total();
        Self {
            slots: (0..num_slots).map(|_| None).collect(),
            zone,
            key_to_slot: HashMap::new(),
            last_used: vec![0u32; num_slots],
            generation: 0,
            slot_to_key: vec![None; num_slots],
            expert_scores: vec![0.0f32; total],
            decode_reuse: vec![0.0f32; total],
            pinned_rows: models
                .iter()
                .map(|(_, s)| pinned_layer_count(s.rows))
                .collect(),
            warm_backed: models
                .iter()
                .map(|(_, s)| Arc::new(HostBacked::new(s.rows, s.experts)))
                .collect(),
            hit_last: vec![false; total],
            models,
            #[cfg(feature = "tensor-assert")]
            owners: None,
        }
    }

    /// Tag every slot with its tenant, and write through every tenant a slot
    /// gains from here on — once the startup fill has installed its experts.
    #[cfg(feature = "tensor-assert")]
    pub(crate) fn attach_owners(&mut self, owners: Arc<SlotOwners>) {
        for (slot, key) in self.slot_to_key.iter().enumerate() {
            if let Some(k) = *key {
                owners.set(slot, k.row, k.expert);
            }
        }
        self.owners = Some(owners);
    }

    /// Slot `slot` gained the tenant `slot_to_key` names: write its tag.
    #[cfg(feature = "tensor-assert")]
    pub(crate) fn mirror_tenant(&self, slot: usize) {
        if let (Some(owners), Some(k)) = (&self.owners, self.slot_to_key[slot]) {
            owners.set(slot, k.row, k.expert);
        }
    }

    /// Record which of `model`'s experts the warm tier holds — each a resident
    /// pinned slot or a pageable copy — once the startup fill is decided; the
    /// stager keeps it current from then on.
    pub(crate) fn set_warm_backed(&mut self, model: ModelIx, membership: &[(usize, usize)]) {
        let backed = &self.warm_backed[model.index()];
        backed.clear();
        for &(row, expert) in membership {
            backed.set(row, expert, true);
        }
    }

    /// `model`'s row `row` was served, routing `routed`: those of its experts
    /// are now the ones it hit on its last visit, and no others.
    pub(crate) fn mark_visit(&mut self, model: ModelIx, row: usize, routed: &[usize]) {
        let span = self.models.row_span(model, row);
        let base = span.start;
        self.hit_last[span].fill(false);
        for &e in routed {
            self.hit_last[base + e] = true;
        }
    }

    /// Whether `key`'s row routed it on its last visit.
    #[inline]
    pub(crate) fn hit_last_visit(&self, key: ExpertKey) -> bool {
        self.hit_last[self.models.flat(key)]
    }

    /// Whether a warm (host) copy backs `key` — its cheap reload.
    #[inline]
    pub(crate) fn is_warm_backed(&self, key: ExpertKey) -> bool {
        self.reload_tier(key) == ReloadTier::Warm
    }

    /// Where `key` reloads from after an eviction — see [`ReloadTier`] for how
    /// eviction weighs it.
    #[inline]
    pub(super) fn reload_tier(&self, key: ExpertKey) -> ReloadTier {
        if self.warm_backed[key.model.index()].get(key.row, key.expert) {
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
    /// `[FADE_FLOOR, PACK_ONLY_PROTECTION]`, so this never divides by zero.
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

    /// Experts of every registered model — each model's every MoE row's full
    /// complement.
    pub(crate) fn total_experts(&self) -> usize {
        self.models.total()
    }

    /// `model`'s MoE rows.
    pub(crate) fn rows(&self, model: ModelIx) -> usize {
        self.models.shape(model).rows
    }

    /// `model`'s experts per MoE row.
    pub(crate) fn experts(&self, model: ModelIx) -> usize {
        self.models.shape(model).experts
    }

    /// `model`'s first rows, which are never evicted ([`pinned_layer_count`]).
    pub(crate) fn pinned_rows(&self, model: ModelIx) -> usize {
        self.pinned_rows[model.index()]
    }

    /// Whether `key` lies in its model's pinned rows: an expert with no copy in
    /// the pack or the warm tier, which no eviction or retraction may drop.
    #[inline]
    pub(crate) fn is_pinned(&self, key: ExpertKey) -> bool {
        key.row < self.pinned_rows[key.model.index()]
    }

    /// Slots every model's pinned rows fill — experts with no copy in the pack
    /// or the warm tier, which no retraction may drop ([`ZoneFloor::Paused`]'s
    /// floor).
    pub(crate) fn pinned_slots(&self) -> usize {
        self.models
            .iter()
            .map(|(m, s)| self.pinned_rows[m.index()] * s.experts)
            .sum()
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
        // Keep key: a pinned-row expert above everything — it has nowhere to
        // reload from, and a zone retracted to the pinned set holds exactly it —
        // then the eviction key without its position: the score scaled by the
        // reload-cost tilt, so a hot warm-backed expert outlasts a pack-only one
        // nobody routes. The same key decides which survivor below the frontier
        // a hotter doomed expert displaces; a pinned-row expert is never
        // displaced.
        let keep: Vec<(bool, f32)> = (0..self.zone.capacity())
            .map(|i| {
                self.slot_to_key[i].map_or((false, 0.0), |k| {
                    (
                        self.is_pinned(k),
                        Self::protected_by(self.score(k), tier_protection(self.reload_tier(k))),
                    )
                })
            })
            .collect();
        let evictable: Vec<bool> = (0..self.zone.capacity())
            .map(|i| self.slot_to_key[i].is_some_and(|k| !self.is_pinned(k)))
            .collect();
        let before = self.zone.capacity();
        let plan = self
            .zone
            .retract_within(target, floor, |i| keep[i], |i| evictable[i]);
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
    /// Returns the evicted expert's key, or `None` if the slot was already
    /// empty.
    ///
    /// **Eviction moves no bytes.** The cold tier holds a valid copy of every
    /// expert at all times, so there is nothing to write back and nowhere to
    /// write it — this used to hand the `ExpertSlot` to the caller for a D2H
    /// copy into pinned RAM, and that copy duplicated data the pack file
    /// already held. Dropping the slot here releases the three `QMatMul` views;
    /// the zone owns the bytes and keeps them.
    pub(crate) fn evict(&mut self, slot_idx: usize) -> Option<ExpertKey> {
        let evicted = self.slot_to_key[slot_idx];
        if let Some(evict_key) = evicted {
            self.key_to_slot.remove(&evict_key);
        }
        self.slot_to_key[slot_idx] = None;
        self.slots[slot_idx] = None;
        evicted
    }

    // ── Score update methods ─────────────────────────────────────────

    /// Get the current score for `key`.
    #[inline]
    pub(crate) fn score(&self, key: ExpertKey) -> f32 {
        let idx = self.models.flat(key);
        self.expert_scores[idx] + DECODE_REUSE_WEIGHT * self.decode_reuse[idx]
    }

    /// Record a decode hit: +1.0, and +1.0 of long-memory decode reuse.
    #[inline]
    pub(crate) fn record_hit(&mut self, key: ExpertKey) {
        let idx = self.models.flat(key);
        self.expert_scores[idx] += 1.0;
        self.decode_reuse[idx] += 1.0;
    }

    /// Record a decode miss: +1.0, so the expert holds its slot to the next
    /// step, which near-certainly routes to it again — but no decode reuse
    /// until it is hit.
    #[inline]
    pub(crate) fn record_decode_miss(&mut self, key: ExpertKey) {
        let idx = self.models.flat(key);
        self.expert_scores[idx] += 1.0;
    }

    /// Record a prefill-attributed hit on an already-resident expert:
    /// [`PREFILL_HIT_SCORE`] (+0.1) rather than the full decode weight.
    #[inline]
    pub(crate) fn record_prefill_hit(&mut self, key: ExpertKey) {
        let idx = self.models.flat(key);
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
    pub(crate) fn record_prefill_elevate(&mut self, key: ExpertKey) {
        let idx = self.models.flat(key);
        self.expert_scores[idx] = PREFILL_ELEVATE_PENALTY;
    }

    /// Record a successful speculative prediction: +0.3.
    #[inline]
    pub(crate) fn record_prediction_hit(&mut self, key: ExpertKey) {
        let idx = self.models.flat(key);
        self.expert_scores[idx] += 0.3;
    }

    /// End-of-pass exponential decay: multiply every model's scores by
    /// `factor` (e.g. 0.85), and decode reuse by [`DECODE_REUSE_DECAY`].
    pub(crate) fn decay_scores(&mut self, factor: f32) {
        for s in self.expert_scores.iter_mut() {
            *s *= factor;
        }
        for r in self.decode_reuse.iter_mut() {
            *r *= DECODE_REUSE_DECAY;
        }
    }

    /// Forward (wrapped) distance from the row `current` is serving to `key`'s
    /// row: how many rows until the wave reaches it again. Distance 0 = the row
    /// being computed right now; distance `n-1` = the row that just executed.
    ///
    /// Defined only within the forwarding model's cycle: a row of any other
    /// model is maximally behind, `n-1` of its own `n` rows, so it gets no
    /// position-based protection.
    #[inline]
    pub(super) fn forward_distance(&self, key: ExpertKey, current: (ModelIx, usize)) -> usize {
        let n = self.rows(key.model);
        let (model, row) = current;
        if key.model != model {
            n.saturating_sub(1)
        } else if key.row >= row {
            key.row - row
        } else {
            n - row + key.row
        }
    }

    /// Up to `count` batch victims, ranked by [`VictimKey::order`] within each
    /// [`RECENCY_PASSES`] pass: a later pass only for what the earlier ones
    /// could not supply. Best victim first.
    ///
    /// `weight(slot_idx, key, dist)` admits a non-pinned occupied slot by
    /// returning the protection factor its frequency is scaled by, or `None` to
    /// skip it. The reload tier's factor ([`tier_protection`]) multiplies in,
    /// and the product goes through [`Self::protected_by`], so an elevated
    /// expert's negative score is not inverted.
    pub(super) fn batch_victims(
        &self,
        current: (ModelIx, usize),
        count: usize,
        weight: impl Fn(usize, ExpertKey, usize) -> Option<f32>,
    ) -> Vec<usize> {
        let mut victims = Vec::with_capacity(count);
        for hit in RECENCY_PASSES {
            let need = count - victims.len();
            if need == 0 {
                break;
            }
            let mut cands: Vec<VictimKey> = self
                .slot_to_key
                .iter()
                .enumerate()
                .filter_map(|(idx, key)| {
                    let k = (*key)?;
                    if self.is_pinned(k) || self.hit_last_visit(k) != hit {
                        return None;
                    }
                    let dist = self.forward_distance(k, current);
                    let tier = self.reload_tier(k);
                    let factor = weight(idx, k, dist)? * tier_protection(tier);
                    Some(VictimKey {
                        slot: idx,
                        score: Self::protected_by(self.score(k), factor),
                        dist,
                        warm: tier == ReloadTier::Warm,
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
    /// `current` is the model forwarding and the row it is serving.
    ///
    /// ## Victim key: `frequency × behind_protection × tier_protection`
    ///
    /// Lowest key first. The protection fades with how far behind the wave a
    /// row is ([`behind_protection`]): L−1 is scaled by [`FADE_FLOOR`] and each
    /// row further back a little more, wrapping round to the rows about to run,
    /// which keep nearly their whole score. So the wave protects what is ahead
    /// of it at the expense of what is behind, and an expert just behind
    /// survives one about to run only with about `1 / FADE_FLOOR` its score —
    /// enough to steer churn behind the wave, not enough to evict a hot expert
    /// the next pass needs for a cold one ahead. Another model's rows are all
    /// maximally behind ([`Self::forward_distance`]).
    ///
    /// A pack-only expert's key is doubled ([`PACK_ONLY_PROTECTION`]) for the
    /// drive read its reload costs. That is a tilt inside the one order, not a
    /// pass above it, so every resident of either tier competes on how much it
    /// is used: a pack-only expert decode no longer routes goes before a
    /// warm-backed one it keeps routing.
    ///
    /// Demand and read-ahead claims spend the same ranked offers, so a read-ahead
    /// copy for a row ahead is bought from the rows the wave has left, not from
    /// the ones it is about to route.
    ///
    /// The policy runs once per recency pass ([`RECENCY_PASSES`]) — the experts
    /// the row did not route on its last visit, then the ones it did — and the
    /// second only for the shortfall.
    ///
    /// Ties break farther-first, then warm-backed, then LRU, so a never-used
    /// expert goes from the row just behind first. Pinned rows are never
    /// candidates. `admit(slot, key)` refuses a slot outright.
    ///
    /// One O(slots) scan + an O(n) `select_nth` partition per recency pass.
    pub(crate) fn rank_victims(
        &self,
        current: (ModelIx, usize),
        count: usize,
        admit: impl Fn(usize, ExpertKey) -> bool,
    ) -> Vec<usize> {
        if count == 0 {
            return Vec::new();
        }
        self.batch_victims(current, count, |idx, key, dist| {
            admit(idx, key).then(|| behind_protection(dist, self.rows(key.model)))
        })
    }

    /// Install an expert into a slot and update all lookup tables.
    pub(crate) fn install(&mut self, slot_idx: usize, key: ExpertKey, slot: ExpertSlot) {
        self.slots[slot_idx] = Some(slot);
        self.key_to_slot.insert(key, slot_idx);
        self.slot_to_key[slot_idx] = Some(key);
        #[cfg(feature = "tensor-assert")]
        self.mirror_tenant(slot_idx);
        self.promote(slot_idx);
    }
}
