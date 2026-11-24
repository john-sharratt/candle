//! Tests that the reload tier tilts eviction between warm-backed and pack-only
//! experts without deciding it.
//!
//! A pack-only expert costs a drive read to bring back, so its key is doubled
//! (`PACK_ONLY_PROTECTION`): at equal use and position the warm-backed one goes
//! first, and among never-used experts the tier breaks the tie. But the tilt sits
//! inside the same least-used order as everything else, so a pack-only expert
//! nobody routes goes before a warm-backed one decode keeps routing, and a batch
//! takes both tiers in one order. With no warm tier at all, the policy is the
//! plain frequency one.

#[cfg(test)]
pub(super) mod tests {

    use super::super::tests_floor::tests::cache;

    use super::super::tests_scores::tests::{best_victim, occupy};

    use super::super::tests_distance::tests::demand_eviction;

    /// Two elevated experts, one warm and one pack-only, through the
    /// single-victim path: dividing the negative score by the pack-only tilt
    /// keeps it the less negative (−0.123 against −0.246), so the warm-backed
    /// one is taken.
    #[test]
    fn a_prefill_elevated_pack_only_expert_is_not_evicted_before_a_warm_one() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        occupy(&mut inner, 1, 10, 101, 2, 0.0);
        inner.record_prefill_elevate(10, 100);
        inner.record_prefill_elevate(10, 101);
        // 100 keeps a warm copy; 101 lives only in the pack, so it is the one
        // worth keeping.
        inner.set_warm_backed(&[(10, 100)]);
        assert_eq!(
            best_victim(&inner, 20, &[]),
            Some((0, (10, 100))),
            "the warm-backed expert is the cheaper victim"
        );
    }

    /// The same in `demand_eviction`.
    #[test]
    fn a_demand_batch_does_not_evict_the_pack_only_elevated_expert_first() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 1, 0.0);
        occupy(&mut inner, 1, 10, 101, 2, 0.0);
        inner.record_prefill_elevate(10, 100);
        inner.record_prefill_elevate(10, 101);
        inner.set_warm_backed(&[(10, 100)]);
        let evicted = demand_eviction(&mut inner, 15, 1, &[]);
        assert_eq!(
            evicted,
            vec![(10, 100)],
            "the warm-backed expert is the cheaper victim"
        );
    }

    /// **Among never-used experts the tier breaks the tie.**
    ///
    /// No factor separates two keys of exactly zero, so the order falls through
    /// to distance, then to the reload tier, before LRU. The LRU order here is
    /// set against the tier on purpose (the pack-only expert is the least
    /// recently used), and the warm-backed one still goes first.
    #[test]
    fn among_never_used_experts_the_pack_only_one_is_not_the_victim() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 5, 0.0); // warm, most recently used
        occupy(&mut inner, 1, 10, 101, 1, 0.0); // pack-only, least recently used
        inner.set_warm_backed(&[(10, 100)]);
        assert_eq!(inner.score(10, 100), 0.0);
        assert_eq!(inner.score(10, 101), 0.0);

        let evicted = demand_eviction(&mut inner, 15, 1, &[]);
        assert_eq!(
            evicted,
            vec![(10, 100)],
            "the reload tier outranks LRU among equal keys"
        );
    }

    /// An expert with no warm copy costs an NVMe read to bring back, so it is
    /// kept in preference to an equally-used one that reloads over PCIe.
    #[test]
    fn the_expert_with_no_warm_copy_is_kept_at_equal_use() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 5, 1.0); // warm-backed
        occupy(&mut inner, 1, 10, 51, 5, 1.0); // pack-only, same temperature
        inner.set_warm_backed(&[(10, 50)]);

        assert_eq!(
            best_victim(&inner, 20, &[]).map(|v| v.1),
            Some((10, 50)),
            "evicted the expert that would have to come back from disk"
        );
    }

    /// **The tier is a tilt, not a pass:** a pack-only expert nobody routes
    /// (0.1 × 0.391 × 2 = 0.078) goes before a hot warm-backed one (40 ×
    /// 0.406).
    #[test]
    fn a_cold_pack_only_expert_goes_before_a_hot_warm_backed_one() {
        let mut inner = cache(5);
        occupy(&mut inner, 0, 10, 50, 1, 40.0); // warm-backed and very hot
        occupy(&mut inner, 1, 11, 51, 2, 0.1); // pack-only and cold
        occupy(&mut inner, 2, 12, 52, 3, 0.2);
        occupy(&mut inner, 3, 13, 53, 4, 0.3);
        occupy(&mut inner, 4, 14, 54, 5, 0.4);
        inner.set_warm_backed(&[(10, 50)]);

        assert_eq!(best_victim(&inner, 20, &[]), Some((1, (11, 51))));
    }

    /// **The startup fill's leftovers turn over.** A pack-only expert the card
    /// took at startup and decode never routed (score zero) goes before
    /// warm-backed experts decode is using, even ahead of the wave.
    #[test]
    fn an_unused_pack_only_expert_goes_before_used_warm_ones() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 50, 1, 0.0); // behind, pack-only, never used
        occupy(&mut inner, 1, 30, 51, 2, 7.0); // ahead, warm-backed, hot
        occupy(&mut inner, 2, 40, 52, 3, 2.0); // ahead, warm-backed, cooler
        inner.set_warm_backed(&[(30, 51), (40, 52)]);
        assert_eq!(
            best_victim(&inner, 20, &[]),
            Some((0, (10, 50))),
            "the never-used pack-only expert"
        );
    }

    /// A batch takes both tiers in one order, each key frequency × distance ×
    /// tilt, at current = 35 of 48: pack 0.5 at 23 behind (0.609), pack 3.0 at
    /// 3 behind (1.781), warm 3.0 at 24 behind (1.875), pack 2.0 at 22 behind
    /// (2.375) — and the warm 6.0 at 25 behind (3.844) survives.
    #[test]
    fn a_demand_batch_blends_both_tiers_in_one_order() {
        let mut inner = cache(5);
        occupy(&mut inner, 0, 10, 50, 1, 6.0); // warm
        occupy(&mut inner, 1, 11, 51, 2, 3.0); // warm
        occupy(&mut inner, 2, 12, 52, 3, 0.5); // pack-only
        occupy(&mut inner, 3, 13, 53, 4, 2.0); // pack-only
        occupy(&mut inner, 4, 32, 54, 5, 3.0); // pack-only
        inner.set_warm_backed(&[(10, 50), (11, 51)]);

        let evicted = demand_eviction(&mut inner, 35, 4, &[]);
        assert_eq!(evicted, vec![(12, 52), (32, 54), (11, 51), (13, 53)]);
        assert!(inner.key_to_slot.contains_key(&(10, 50)));
        assert_eq!(inner.free_len(), 4);
    }

    /// With no warm tier every expert is pack-only, every key is doubled alike,
    /// and the order is the plain policy's: the same victims in the same order.
    #[test]
    fn with_no_warm_tier_the_policy_is_unchanged() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 50, 1, 6.0);
        occupy(&mut inner, 1, 20, 51, 2, 0.5);
        occupy(&mut inner, 2, 33, 52, 3, 0.8); // L-2: key 0.225, under 20's 0.242
        let evicted = demand_eviction(&mut inner, 35, 2, &[]);
        assert_eq!(evicted, vec![(33, 52), (20, 51)]);
    }
}
