//! Tests that the reload tier, rather than the score, decides eviction between
//! warm-backed and pack-only experts.
//!
//! A warm-backed expert has a cheap host copy to reload from, so the warm pass
//! runs before the pack pass whatever the two scores say, and a pack-only expert
//! is kept in preference to an equally cold warm one. The batch path takes the
//! warm pass first and runs the pack pass only for the shortfall. With no warm
//! tier at all, the passes collapse to the plain frequency policy.

#[cfg(test)]
pub(super) mod tests {

    use super::super::tests_floor::tests::cache;

    use super::super::tests_scores::tests::{best_victim, occupy};

    use super::super::tests_distance::tests::demand_eviction;

    /// Two elevated experts, one warm and one pack-only, through the
    /// single-victim path: the warm pass runs first, so the warm-backed one is
    /// taken however the two scores compare — the elevation cannot pull the
    /// pack-only expert ahead of its pass.
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

    /// The same in `demand_eviction`: the elevated pack-only expert is in the
    /// second pass, which is never reached while the warm pass can supply.
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

    /// **The cold shield reaches the never-used group, where every key is
    /// exactly zero.**
    ///
    /// No score can separate two never-used experts, so within one pass the
    /// scan falls through to LRU — which knows nothing about what a reload
    /// costs. The tier pass is what keeps the shield: the LRU order here is set
    /// against it on purpose (the pack-only expert is the least recently used),
    /// and the warm-backed one still goes first because its pass runs first.
    ///
    /// Never-used residents are not an edge case: a decode-attributed elevation
    /// records no score at all, and a speculative prefetch's score stays at zero
    /// until its prediction validates.
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
            "the reload tier outranks LRU among equal scores"
        );
    }

    /// An expert with no warm copy costs an NVMe read to bring back, so it is
    /// kept in preference to an equally-cold one that reloads over PCIe.
    #[test]
    fn the_expert_with_no_warm_copy_is_kept() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 5, 1.0); // warm-backed
        occupy(&mut inner, 1, 10, 51, 5, 1.0); // cold-only, same temperature
        inner.set_warm_backed(&[(10, 50)]);

        assert_eq!(
            best_victim(&inner, 20, &[]).map(|v| v.1),
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

        assert_eq!(best_victim(&inner, 20, &[]), Some((0, (10, 50))));
    }

    /// The warm-backed experts are exhausted before a pack-only one is
    /// considered, so a warm-backed expert AHEAD of the wave goes before a
    /// pack-only one behind it.
    #[test]
    fn the_warm_pass_is_exhausted_before_the_pack() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 50, 1, 0.0); // behind, pack-only, never used
        occupy(&mut inner, 1, 30, 51, 2, 7.0); // ahead, warm-backed, hot
        occupy(&mut inner, 2, 40, 52, 3, 2.0); // ahead, warm-backed, cooler
        inner.set_warm_backed(&[(30, 51), (40, 52)]);
        assert_eq!(
            best_victim(&inner, 20, &[]),
            Some((2, (40, 52))),
            "coolest warm-backed expert"
        );
    }

    /// The batch path takes what the warm pass can supply, then runs the same
    /// policy over the pack-only experts for the shortfall — each pass in its
    /// own frequency × protection order.
    #[test]
    fn demand_eviction_runs_the_pack_pass_only_for_the_shortfall() {
        let mut inner = cache(5);
        occupy(&mut inner, 0, 10, 50, 1, 6.0); // warm, 25 behind: 3.84
        occupy(&mut inner, 1, 11, 51, 2, 3.0); // warm, 24 behind: 1.875
        occupy(&mut inner, 2, 12, 52, 3, 0.5); // pack-only, 23 behind: 0.305
        occupy(&mut inner, 3, 13, 53, 4, 2.0); // pack-only, 22 behind: 1.19
        occupy(&mut inner, 4, 32, 54, 5, 3.0); // pack-only, 3 behind: 0.891
        inner.set_warm_backed(&[(10, 50), (11, 51)]);

        let evicted = demand_eviction(&mut inner, 35, 4, &[]);
        assert_eq!(
            evicted,
            vec![(11, 51), (10, 50), (12, 52), (32, 54)],
            "both warm (by key), then the two best pack-only (0.305, then 0.891)"
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
        occupy(&mut inner, 2, 33, 52, 3, 0.8); // L-2: key 0.225, under 20's 0.242
        let evicted = demand_eviction(&mut inner, 35, 2, &[]);
        assert_eq!(evicted, vec![(33, 52), (20, 51)]);
    }
}
