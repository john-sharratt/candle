//! Tests of the score updates and the protection arithmetic behind the eviction
//! key.
//!
//! A decode hit adds the full credit and the decode reuse, a prefill hit adds
//! only the small prefill credit, and a prefill elevation assigns a negative
//! score rather than adding to whatever the slot's previous tenant left behind.
//! Protection is checked to fade from the layer just behind the wave, and to be
//! divided below zero, so an elevated expert ahead of the wave is not inverted
//! onto the one about to run.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_floor::tests::cache;

    use super::super::tests_distance::tests::demand_eviction;

    /// Mark a slot occupied by `(layer, expert)` without a real `ExpertSlot`
    /// (eviction selection reads only the bookkeeping tables, never the VRAM
    /// buffers).
    pub(in super::super) fn occupy(
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

    /// The single best demand victim, `(slot, key)`, if there is one.
    pub(in super::super) fn best_victim(
        inner: &ExpertCacheInner,
        current: usize,
        protect: &[usize],
    ) -> Option<(usize, (usize, usize))> {
        let s = *inner
            .rank_victims(current, 1, |s, _| !protect.contains(&s))
            .first()?;
        Some((
            s,
            inner.slot_to_key[s].expect("a ranked victim holds an expert"),
        ))
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

        assert_eq!(best_victim(&inner, 20, &[]), Some((2, (10, 102))));
    }

    #[test]
    fn record_prefill_hit_bumps_by_the_small_credit_only() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_prefill_hit(10, 100);
        assert_eq!(inner.score(10, 100), 0.1);
        inner.record_prefill_hit(10, 100);
        assert_eq!(inner.score(10, 100), 0.2);
    }

    #[test]
    fn record_hit_still_bumps_by_the_full_decode_credit() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_hit(10, 100);
        let idx = 10 * inner.experts_per_layer + 100;
        assert_eq!(inner.expert_scores[idx], 1.0);
        assert_eq!(inner.decode_reuse[idx], 1.0);
        assert_eq!(inner.score(10, 100), 1.1);
    }

    #[test]
    fn a_decode_miss_earns_the_decode_credit_but_no_reuse() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_decode_miss(10, 100);
        let idx = 10 * inner.experts_per_layer + 100;
        assert_eq!(inner.expert_scores[idx], 1.0);
        assert_eq!(inner.decode_reuse[idx], 0.0);
        assert_eq!(inner.score(10, 100), 1.0);
    }

    #[test]
    fn decode_reuse_decays_on_its_own_slower_clock() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_hit(10, 100);
        inner.decay_scores(0.5);
        let idx = 10 * inner.experts_per_layer + 100;
        assert_eq!(inner.expert_scores[idx], 0.5);
        assert_eq!(inner.decode_reuse[idx], 0.99);
    }

    /// Two experts with the same recency score: the one decode has hit outranks
    /// the one it has only missed.
    #[test]
    fn a_one_off_decode_miss_is_evicted_before_a_reused_expert() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        occupy(&mut inner, 1, 10, 101, 1, 0.0);
        inner.record_hit(10, 100);
        inner.record_decode_miss(10, 101);
        assert_eq!(best_victim(&inner, 20, &[]), Some((1, (10, 101))));
    }

    #[test]
    fn record_prefill_elevate_pushes_the_score_negative() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_prefill_elevate(10, 100);
        assert_eq!(inner.score(10, 100), -0.1);
    }

    /// An elevation resets the recency score but keeps the decode reuse the
    /// same expert earned: two decode hits leave −0.1 + 0.1 × 2.
    #[test]
    fn a_prefill_elevation_keeps_the_experts_decode_reuse() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        inner.record_hit(10, 100);
        inner.record_hit(10, 100);
        inner.record_prefill_elevate(10, 100);
        let idx = inner.score_idx(10, 100);
        assert_eq!(inner.expert_scores[idx], -0.1);
        assert_eq!(inner.decode_reuse[idx], 2.0);
        assert_eq!(inner.score(10, 100), 0.1);
    }

    /// **A previous tenant's score is not inherited.** `install` leaves
    /// `expert_scores` alone, so a slot that held a decode-hot expert still
    /// carries its score when a prefill miss lands a different expert on it.
    /// Adding the penalty there would leave the newcomer protected at 11.9;
    /// assigning it states what an elevation actually means.
    #[test]
    fn a_prefill_elevation_does_not_inherit_a_hot_predecessors_score() {
        let mut inner = cache(1);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        let idx = inner.score_idx(10, 100);
        inner.expert_scores[idx] = 12.0;
        inner.record_prefill_elevate(10, 100);
        assert_eq!(inner.score(10, 100), -0.1);
    }

    /// The protection, raw, for 48 layers: the floor plus 0.75 × 1/48
    /// (= 1/64) per layer behind, from L−1 through the wrap to the layer being
    /// computed.
    #[test]
    fn protection_fades_from_the_layer_just_behind() {
        let n = 48;
        assert_eq!(behind_protection(47, n), 0.265625, "L-1");
        assert_eq!(behind_protection(45, n), 0.296875, "L-3");
        assert_eq!(behind_protection(24, n), 0.625, "half a pass behind");
        assert_eq!(behind_protection(1, n), 0.984375, "L+1");
        assert_eq!(behind_protection(0, n), 1.0, "the layer being computed");
    }

    /// **A prefill-lifted expert is evicted before any decode-scored one of its
    /// reload tier, at every position.** This is what keeps the cache hot for
    /// decode: a prefill sweep touches most of the table roughly once, so what
    /// it drags in must not displace the working set decode actually reuses.
    /// Checked at the extremes of the protection rather than at one convenient
    /// layer, because the ordering has to hold across the whole range — the
    /// least-evictable prefill lift (the layer being computed) against the
    /// most-evictable decode hit (L−1).
    #[test]
    fn a_prefill_lift_sorts_below_every_decode_hit_at_both_extremes() {
        let n = 48usize;
        let worst_protect = behind_protection(n - 1, n); // L-1
        let best_protect = behind_protection(0, n); // the layer being computed
        assert_eq!((worst_protect, best_protect), (0.265625, 1.0));
        // The least-evictable a prefill lift can be, and the most-evictable a
        // decode hit can be.
        let softest_lift = PREFILL_ELEVATE_PENALTY / best_protect;
        let weakest_decode = 1.0f32 * worst_protect;
        assert!(
            softest_lift < weakest_decode,
            "a prefill lift ({softest_lift}) must still evict before the weakest \
             decode hit ({weakest_decode})"
        );
        // And below a prefill HIT, which is the nearest positive neighbour.
        let weakest_prefill_hit = PREFILL_HIT_SCORE * worst_protect;
        assert!(
            softest_lift < weakest_prefill_hit,
            "a prefill lift ({softest_lift}) must evict before a prefill hit \
             ({weakest_prefill_hit})"
        );
    }

    /// **The protection must not invert an elevated score onto a layer ahead.**
    /// Two elevated warm-backed experts: layer 10 is five layers behind the wave
    /// (protection 0.328) and layer 20 five ahead of it (0.922). Multiplying
    /// would give the one behind −0.033 and the one ahead −0.092, evicting the
    /// expert the wave is about to route to and keeping the one it has just
    /// left; dividing gives −0.305 and −0.108.
    #[test]
    fn the_protection_does_not_invert_an_elevated_score_onto_an_upcoming_layer() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 1, 0.0); // dist 43 → five behind
        occupy(&mut inner, 1, 20, 101, 2, 0.0); // dist 5 → about to be used
        inner.record_prefill_elevate(10, 100);
        inner.record_prefill_elevate(20, 101);
        inner.set_warm_backed(&[(10, 100), (20, 101)]);
        let evicted = demand_eviction(&mut inner, 15, 1, &[]);
        assert_eq!(
            evicted,
            vec![(10, 100)],
            "the behind-the-wave expert goes, not the one five layers ahead"
        );
    }

    /// **The sign stays a hard tier within a pass in the batch paths too.** An
    /// elevated expert ahead of the wave must still be taken before one with a
    /// single decode hit behind it: −0.1 / 0.922 against 1.0 × 0.328. This is
    /// what `protected_by`'s divide preserves and what a shift-then-multiply
    /// scheme would lose. Both are pack-only, so the two share a pass and only
    /// the key decides.
    #[test]
    fn an_elevated_expert_outranks_nothing_scored_in_a_demand_batch() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 20, 100, 1, 0.0); // five ahead of the wave
        occupy(&mut inner, 1, 10, 101, 2, 1.0); // one decode hit, five behind
        inner.record_prefill_elevate(20, 100);
        let evicted = demand_eviction(&mut inner, 15, 1, &[]);
        assert_eq!(
            evicted,
            vec![(20, 100)],
            "an elevated expert is evicted before any scored one"
        );
    }

    /// **A prefill-elevated expert is evicted before any hit, decode or
    /// prefill, once the pass has moved past its layer.** This is the whole
    /// point of the negative score: a fresh prefill load at layer 10 must not
    /// out-survive genuinely reused experts once the sweep is at layer 20.
    #[test]
    fn prefill_elevated_expert_is_the_preferred_victim_over_any_hit() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 100, 1, 0.0); // freshly prefill-elevated
        inner.record_prefill_elevate(10, 100);
        occupy(&mut inner, 1, 10, 101, 2, 0.1); // one prefill hit
        occupy(&mut inner, 2, 10, 102, 3, 1.0); // one decode hit

        assert_eq!(best_victim(&inner, 20, &[]), Some((0, (10, 100))));
    }

    #[test]
    fn hot_expert_survives_a_cold_one() {
        // Same layer (same protection): the frequently-used expert is kept
        // and the cold one evicted — the cache is frequency-dominated.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 9, 9.0);
        occupy(&mut inner, 1, 10, 51, 1, 0.0);

        assert_eq!(
            best_victim(&inner, 20, &[]).map(|v| v.1),
            Some((10, 51)),
            "cold expert was not evicted in preference to the hot one"
        );
    }
}
