//! Tests that the zone floor always leaves the eviction scan a victim, and that
//! a concession takes the right experts.
//!
//! A cache at or above the floor must always have an evictable slot, because a
//! cache that fills with pinned-layer experts can never free one again. The
//! pinned depth is checked to stay fixed at every capacity. A concession is
//! checked to drop warm-backed experts before pack-only ones, to keep exactly
//! the pinned set while the engine is paused, and to displace the coldest
//! unpinned survivor when a full zone gives up a hot doomed expert.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_scores::tests::{best_victim, occupy};

    /// A cache of `n` slots over a zone with no device behind it.
    ///
    /// The zone is pure arithmetic — addresses and indices — so the whole
    /// eviction policy still exercises with no GPU and no model load, exactly as
    /// it did when the free list was a local `Vec`.
    /// Pins the standard head layers, as every cache now does.
    pub(in super::super) fn cache(n: usize) -> ExpertCacheInner {
        ExpertCacheInner::new(WeightZone::new(1 << 30, 4096, n, n, 0), 48, 128)
    }

    /// A cache sized as the loader would size it.
    pub(in super::super) fn sized_cache(n: usize, experts_per_layer: usize) -> ExpertCacheInner {
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
    /// filter matches nothing, and no promotion can take a slot from then on —
    /// permanently, because escaping the state requires an eviction the state
    /// forbids. The
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
                    best_victim(&inner, capacity / experts_per_layer, &[]).is_some(),
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
    /// pinned-layer filter matches nothing and no promotion can take a slot from
    /// then on, permanently, because escaping the state requires an eviction the state
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
            best_victim(&inner, PINNED_LAYERS, &[]).is_some(),
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
            best_victim(&starved, PINNED_LAYERS, &[]).is_none(),
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

        let plan = inner.retract_zone(4, ZoneFloor::Serving);
        assert_eq!(plan.relocate, vec![(5, 0)]);
        assert_eq!(plan.evict, vec![4]);
    }

    /// **Paused, the zone goes to the pinned set, and every pinned expert
    /// survives** however cold, displacing hotter unpinned survivors: it has no
    /// copy to come back from. Serving, the same ask stops at the floor.
    #[test]
    fn a_paused_concession_keeps_exactly_the_pinned_set() {
        let mut inner = ExpertCacheInner::new(WeightZone::new(1 << 30, 4096, 6, 6, 4), 48, 1);
        let p = inner.pinned_layers;
        assert_eq!(inner.pinned_slots(), p);
        assert_eq!(p, 2, "the fixture prices two pinned layers");
        for slot in 0..4 {
            occupy(&mut inner, slot, p + slot, 0, slot as u32, 9.0); // hot, unpinned
        }
        occupy(&mut inner, 4, 0, 0, 4, 0.0); // pinned, doomed, coldest of all
        occupy(&mut inner, 5, 1, 0, 5, 0.0); // pinned, doomed

        let serving = inner.retract_zone(2, ZoneFloor::Serving);
        assert_eq!(inner.zone.capacity(), 4, "serving, the stated floor holds");
        assert!(serving.evict.iter().all(|&s| s >= 4));

        let mut inner = ExpertCacheInner::new(WeightZone::new(1 << 30, 4096, 6, 6, 4), 48, 1);
        for slot in 0..4 {
            occupy(&mut inner, slot, p + slot, 0, slot as u32, 9.0);
        }
        occupy(&mut inner, 4, 0, 0, 4, 0.0);
        occupy(&mut inner, 5, 1, 0, 5, 0.0);
        let plan = inner.retract_zone(0, ZoneFloor::Paused);
        assert_eq!(
            inner.zone.capacity(),
            2,
            "paused, the pinned set is the floor"
        );
        let kept: Vec<usize> = plan.relocate.iter().map(|&(from, _)| from).collect();
        assert!(
            kept.contains(&4) && kept.contains(&5),
            "a pinned expert was dropped: relocate {:?}, evict {:?}",
            plan.relocate,
            plan.evict
        );
        assert!(!plan.evict.contains(&4) && !plan.evict.contains(&5));
        assert_eq!(inner.zone.stated_floor(), 4, "the floor to grow back to");
    }

    /// A concession on a full zone trades a hot doomed expert for the coldest
    /// survivor below the frontier — never for a pinned-layer expert, which has
    /// no tier to reload from, however cold.
    #[test]
    fn a_full_concession_displaces_the_coldest_unpinned_survivor() {
        let mut inner = sized_cache(6, 1);
        let p = inner.pinned_layers;
        assert!(p >= 1, "the fixture needs a pinned layer");
        occupy(&mut inner, 0, 0, 0, 0, 0.0); // pinned, coldest of all
        occupy(&mut inner, 1, p + 1, 0, 1, 0.1); // the coldest unpinned survivor
        occupy(&mut inner, 2, p + 2, 0, 2, 5.0);
        occupy(&mut inner, 3, p + 3, 0, 3, 5.0);
        occupy(&mut inner, 4, p + 4, 0, 4, 9.0); // doomed, hot
        occupy(&mut inner, 5, p + 5, 0, 5, 0.05); // doomed, cold

        let plan = inner.retract_zone(4, ZoneFloor::Serving);
        assert_eq!(plan.displace, vec![1]);
        assert_eq!(plan.relocate, vec![(4, 1)]);
        assert_eq!(plan.evict, vec![5]);
    }
}
