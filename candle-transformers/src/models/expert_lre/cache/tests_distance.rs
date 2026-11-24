//! Tests of the victim order by distance behind the wave, reload tier and the
//! last visit's routing.
//!
//! Each case fixes the wave's position and the scores, then checks which expert
//! the ranking takes first. The layers just behind the wave go before those
//! further back, distance outweighs the small reload-cost tilt, and an expert
//! its row routed on the last visit goes only after every stale one. A
//! pinned-layer expert is never a candidate, and ranking alone evicts nothing.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_floor::tests::cache;

    use super::super::tests_scores::tests::{best_victim, occupy};

    /// The demand-driven choice with every candidate admissible but `protect`:
    /// rank, then evict and free each victim. Returns the evicted keys.
    pub(in super::super) fn demand_eviction(
        inner: &mut ExpertCacheInner,
        current: usize,
        count: usize,
        protect: &[usize],
    ) -> Vec<(usize, usize)> {
        let victims = inner.rank_victims(current, count, |s, _| !protect.contains(&s));
        victims
            .into_iter()
            .map(|s| {
                let key = inner.evict(s).expect("a ranked victim holds an expert");
                inner.zone.release(s);
                key
            })
            .collect()
    }

    #[test]
    fn demand_eviction_prefers_the_layers_just_behind() {
        // current=35, n=48. Layer 30 is five behind (key 0.5 × 0.328 = 0.164),
        // layer 20 fifteen behind (0.4 × 0.484 = 0.194): the warmer expert just
        // behind goes BEFORE the colder one further back, steering churn onto
        // the layers the wave just left.
        let mut inner = cache(4);
        occupy(&mut inner, 0, 1, 100, 1, 0.1); // pinned (layer < PINNED_LAYERS)
        occupy(&mut inner, 1, 10, 101, 2, 5.0);
        occupy(&mut inner, 2, 20, 102, 3, 0.4); // colder, further back
        occupy(&mut inner, 3, 30, 103, 4, 0.5); // just behind → victim

        let evicted = demand_eviction(&mut inner, 35, 1, &[]);
        assert_eq!(evicted, vec![(30, 103)], "just-behind expert goes first");
        assert!(inner.key_to_slot.contains_key(&(10, 101)));
        assert!(inner.key_to_slot.contains_key(&(20, 102)));
        assert!(
            inner.key_to_slot.contains_key(&(1, 100)),
            "pinned layer was evicted"
        );
        assert_eq!(inner.free_len(), 1, "exactly the demanded count freed");
    }

    #[test]
    fn demand_eviction_lets_frequency_outweigh_a_few_layers() {
        // Three candidates just behind current=35: a cold expert four back
        // (0.3 × 0.3125 = 0.094) goes before a warm one at L-1 (2.0 × 0.266 =
        // 0.53) and a hot one at L-2 (6.0 × 0.281 = 1.69).
        let mut inner = cache(3);
        occupy(&mut inner, 0, 33, 100, 1, 6.0); // L-2, hot
        occupy(&mut inner, 1, 31, 101, 2, 0.3); // L-4, coldest → victim
        occupy(&mut inner, 2, 34, 102, 3, 2.0); // L-1, warm
        let evicted = demand_eviction(&mut inner, 35, 1, &[]);
        assert_eq!(evicted, vec![(31, 101)]);
    }

    #[test]
    fn distance_outweighs_the_reload_tilt() {
        // Equally used, the pack-only expert just behind the wave (1.0 × 0.297
        // × 2 = 0.594) goes before the warm-backed one about to run (1.0 ×
        // 0.922): the tilt halves the gap a pack read costs, it does not
        // overrule the layer the wave needs next.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 32, 100, 1, 1.0); // just behind, pack-only
        occupy(&mut inner, 1, 40, 101, 2, 1.0); // ahead, warm-backed
        inner.set_warm_backed(&[(40, 101)]);
        let evicted = demand_eviction(&mut inner, 35, 1, &[]);
        assert_eq!(evicted, vec![(32, 100)], "the expert just behind goes");
        assert!(inner.key_to_slot.contains_key(&(40, 101)));
    }

    #[test]
    fn demand_eviction_protects_the_layer_hits() {
        // A protected slot (the caller lists the current layer's hits) is
        // spared even when it is the best victim on the card; the eviction
        // takes the next one instead.
        let mut inner = cache(3);
        occupy(&mut inner, 0, 30, 100, 1, 0.1); // best victim — but a HIT, protected
        occupy(&mut inner, 1, 10, 101, 2, 5.0);
        occupy(&mut inner, 2, 33, 102, 3, 0.4); // L-2, next best → victim
        let evicted = demand_eviction(&mut inner, 35, 1, &[0]);
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
        occupy(&mut inner, 1, 20, 101, 2, 0.5); // fifteen behind → victim
        occupy(&mut inner, 2, 37, 102, 3, 4.0);
        let evicted = demand_eviction(&mut inner, 35, 1, &[0]);
        assert_eq!(evicted, vec![(20, 101)], "in-flight install spared");
        assert!(inner.key_to_slot.contains_key(&(36, 100)));
    }

    #[test]
    fn demand_eviction_caps_at_the_candidates() {
        // Asking for more than the non-pinned population frees what exists and
        // no more; a promotion short of victims takes fewer slots.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 1, 100, 1, 0.1); // pinned
        occupy(&mut inner, 1, 10, 101, 2, 0.2);
        let evicted = demand_eviction(&mut inner, 20, 5, &[]);
        assert_eq!(evicted, vec![(10, 101)]);
        assert!(inner.key_to_slot.contains_key(&(1, 100)), "pinned survives");
    }

    #[test]
    fn a_refused_slot_is_never_ranked() {
        // A slot the caller refuses — a row still being computed, which cannot
        // give its slot up yet — is skipped even when it is the lowest-scored
        // slot on the card.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 36, 100, 1, 0.0); // refused
        occupy(&mut inner, 1, 40, 101, 2, 9.0); // hot, but the only legal victim
        assert_eq!(best_victim(&inner, 35, &[0]), Some((1, (40, 101))));
    }

    #[test]
    fn among_equals_ahead_the_furthest_future_goes() {
        // Everything resident is ahead of the wave, equal frequency: the
        // FURTHEST-future expert goes (next use latest — Belady), not the one
        // about to be routed.
        let mut inner = cache(2);
        occupy(&mut inner, 0, 36, 100, 1, 2.0); // L+1 — about to be routed, kept
        occupy(&mut inner, 1, 45, 101, 2, 2.0); // L+10 — furthest future → victim
        assert_eq!(best_victim(&inner, 35, &[]), Some((1, (45, 101))));
    }

    /// **At equal scores the order is the walk backwards from the wave**: L−1,
    /// then L−3, then L+1 — the layer about to run last of all.
    #[test]
    fn equal_scores_go_from_the_layer_just_behind_backwards() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 11, 100, 1, 1.0); // L+1
        occupy(&mut inner, 1, 7, 101, 2, 1.0); // L-3
        occupy(&mut inner, 2, 9, 102, 3, 1.0); // L-1
        assert_eq!(inner.rank_victims(10, 3, |_, _| true), vec![2, 1, 0]);
    }

    /// **The eviction front moves with the wave, through the wrap.** At the
    /// start of a pass (current=2 of 48) the layers just behind are the
    /// previous pass's last: layer 47 (three behind) goes before layer 10
    /// (forty behind — about to run). Well into the pass (current=30) the order
    /// turns over: layer 10 is twenty behind and layer 47 seventeen ahead.
    #[test]
    fn the_eviction_front_wraps_with_the_wave() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 47, 100, 1, 1.0);
        occupy(&mut inner, 1, 10, 101, 2, 1.0);
        assert_eq!(inner.rank_victims(2, 2, |_, _| true), vec![0, 1]);
        assert_eq!(inner.rank_victims(30, 2, |_, _| true), vec![1, 0]);
    }

    /// **Distance steers, frequency decides.** A hot expert just behind the
    /// wave (9.0 × 0.266 = 2.39) outlasts a near-cold one about to run (0.2 ×
    /// 0.984 = 0.197): the next pass needs the hot one. Between two experts of
    /// like temperature the one just behind goes — 1.0 × 0.266 against 0.3 ×
    /// 0.984 = 0.295 — and a never-used one goes first wherever it sits.
    #[test]
    fn frequency_ranks_within_the_fade() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 29, 100, 1, 9.0); // L-1, hot
        occupy(&mut inner, 1, 31, 101, 2, 0.2); // L+1, near cold
        assert_eq!(inner.rank_victims(30, 1, |_, _| true), vec![1]);

        let mut inner = cache(2);
        occupy(&mut inner, 0, 29, 100, 1, 1.0); // L-1
        occupy(&mut inner, 1, 31, 101, 2, 0.3); // L+1
        assert_eq!(inner.rank_victims(30, 1, |_, _| true), vec![0]);

        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 100, 1, 4.0); // L-20, hot
        occupy(&mut inner, 1, 31, 101, 2, 0.0); // L+1, never used
        assert_eq!(inner.rank_victims(30, 1, |_, _| true), vec![1]);
    }

    #[test]
    fn pinned_layers_never_evicted() {
        let mut inner = cache(PINNED_LAYERS);
        for layer in 0..PINNED_LAYERS {
            occupy(&mut inner, layer, layer, 100 + layer, layer as u32 + 1, 0.0);
        }
        // Every resident expert is pinned → no legal victim.
        assert!(best_victim(&inner, 5, &[]).is_none());
    }

    /// **An expert its row just routed goes only after every one it did not,**
    /// whatever their scores: here a just-hit expert scored 1.0 outlasts a
    /// stale one at 3.0, until the row's next visit routes it no more.
    #[test]
    fn a_just_hit_expert_outlasts_every_stale_one() {
        let mut inner = cache(3);
        occupy(&mut inner, 0, 10, 50, 1, 1.0); // just hit
        occupy(&mut inner, 1, 10, 51, 2, 3.0); // stale, hot
        occupy(&mut inner, 2, 20, 52, 3, 5.0); // stale, hotter
        inner.mark_visit(10, &[50]);
        assert_eq!(inner.rank_victims(30, 3, |_, _| true), vec![1, 2, 0]);
        inner.mark_visit(10, &[51]);
        assert_eq!(inner.rank_victims(30, 1, |_, _| true), vec![0]);
    }

    /// Recency runs above the reload tier: a pack-only stale expert goes before
    /// a warm-backed one its row just routed.
    #[test]
    fn recency_outranks_the_reload_tier() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 1, 1.0); // warm, just hit
        occupy(&mut inner, 1, 10, 51, 2, 0.0); // pack-only, stale
        inner.set_warm_backed(&[(10, 50)]);
        inner.mark_visit(10, &[50]);
        assert_eq!(inner.rank_victims(30, 2, |_, _| true), vec![1, 0]);
    }

    /// Within each recency pass both tiers share one order: of four equally
    /// unused experts in one layer, the stale pair goes before the just-hit
    /// pair, and within each pair the warm-backed one goes first on the tie.
    #[test]
    fn each_recency_pass_spans_both_tiers() {
        let mut inner = cache(4);
        occupy(&mut inner, 0, 10, 50, 1, 0.0); // warm, stale
        occupy(&mut inner, 1, 10, 51, 2, 0.0); // warm, just hit
        occupy(&mut inner, 2, 10, 52, 3, 0.0); // pack, stale
        occupy(&mut inner, 3, 10, 53, 4, 0.0); // pack, just hit
        inner.set_warm_backed(&[(10, 50), (10, 51)]);
        inner.mark_visit(10, &[51, 53]);
        assert_eq!(inner.rank_victims(30, 4, |_, _| true), vec![0, 2, 1, 3]);
    }

    /// A visit marks exactly the experts it routed in its row, clearing the
    /// row's previous visit and leaving other rows alone.
    #[test]
    fn a_visit_marks_its_routed_experts_and_clears_the_rest() {
        let mut inner = cache(1);
        inner.mark_visit(10, &[3, 7]);
        inner.mark_visit(11, &[3]);
        inner.mark_visit(10, &[7, 9]);
        let row10: Vec<usize> = (0..12).filter(|&e| inner.hit_last_visit(10, e)).collect();
        let row11: Vec<usize> = (0..12).filter(|&e| inner.hit_last_visit(11, e)).collect();
        assert_eq!((row10, row11), (vec![7, 9], vec![3]));
    }

    /// Ranking evicts nothing: the caller decides per victim.
    #[test]
    fn ranking_leaves_the_tables_alone() {
        let mut inner = cache(2);
        occupy(&mut inner, 0, 10, 50, 5, 5.0);
        occupy(&mut inner, 1, 30, 51, 0, 0.0);
        assert_eq!(inner.rank_victims(20, 2, |_, _| true), vec![1, 0]);
        assert!(inner.key_to_slot.contains_key(&(10, 50)));
        assert!(inner.key_to_slot.contains_key(&(30, 51)));
        assert_eq!(inner.free_len(), 0);
    }
}
