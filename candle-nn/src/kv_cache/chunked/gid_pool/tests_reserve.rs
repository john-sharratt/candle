//! Tests for the reserve each class keeps and which empty arena the sweep takes.
//!
//! A reserve class keeps one arena's worth of free slots at its lowest place, so
//! the sweep releases from the top of the span down. A reserve does not hold the
//! frontier while a hole lies below it; it moves down to the hole and stays there.
//! A class with no reserve, or a cleared one, is swept like any other.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{
        float_key, reversed_pair, standing_arena, test_arena_chunks,
    };

    /// **A reserve class keeps one arena's worth free, the lowest standing**
    /// (`docs/kv_drain_under_pressure.md` §2): of two empty arenas the sweep
    /// takes the higher and keeps the lower, then takes nothing more; a class
    /// with no reserve is swept bare.
    #[test]
    fn a_reserve_class_keeps_one_arenas_worth_free_at_its_lowest() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let other = ArenaKey::new(SizeClass::at(1), ArenaLocation::Gpu);
        pool.keep_reserve(key);
        let low = standing_arena(&pool, key);
        let high = standing_arena(&pool, key);
        assert!(low < high);
        assert_eq!(
            pool.next_tombstone(key, None),
            Some(high),
            "the higher goes first"
        );
        pool.recycle_arena_index(high);
        assert_eq!(pool.next_tombstone(key, None), None, "the reserve stays");
        assert_eq!(pool.free_slots(key), test_arena_chunks());

        let a = standing_arena(&pool, other);
        let b = standing_arena(&pool, other);
        assert_eq!(pool.next_tombstone(other, None), Some(a.max(b)));
        assert_eq!(
            pool.next_tombstone(other, None),
            Some(a.min(b)),
            "no reserve, swept bare"
        );
    }

    /// **The reserve that stays is the lowest in the span, whatever the
    /// indices say**: of two empty arenas the sweep releases the one at the
    /// greater offset into the reservation. Chosen by index, it released the
    /// lower arena and kept the higher one standing on the frontier — region
    /// 3127 on BlackWorld (2026-11-08), for hours.
    #[test]
    fn the_reserve_keeps_the_lowest_placed_arena_whatever_its_index() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.keep_reserve(key);
        let (high, low) = reversed_pair(&pool, key);
        assert_eq!(
            pool.next_tombstone(key, None),
            Some(high),
            "the arena higher in the span goes"
        );
        pool.recycle_arena_index(high);
        assert_eq!(pool.next_tombstone(key, None), None, "the lower one stays");
        assert_eq!(pool.pool_arena_load(key)[0].0, low);
    }

    /// **A reserve does not hold the top while a hole lies below it.** The one
    /// empty arena a reserve keeps is released when a free region stands below
    /// it, so the reserve is made again at the lowest hole; with no hole below,
    /// or one above, it stays.
    #[test]
    fn a_reserve_above_a_hole_is_released_to_move_down() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.keep_reserve(key);
        let only = standing_arena(&pool, key);
        pool.set_arena_rank(key, only, 487);
        assert_eq!(pool.next_tombstone(key, None), None, "no regions: it stays");
        assert_eq!(
            pool.next_tombstone(key, Some(600)),
            None,
            "the only hole is above it: it stays"
        );
        assert_eq!(
            pool.next_tombstone(key, Some(57)),
            Some(only),
            "a hole below: it goes, to be made again there"
        );
    }

    /// **A moved reserve settles: it does not churn at the bottom.** Released
    /// from region 487 over a hole at 57, the reserve is made again at 57 — the
    /// lowest free region a claim takes — and the next sweep, whose lowest hole
    /// is now the region it left, keeps it. Nothing below it is free, so nothing
    /// releases it again: the free-and-remake loop at the bottom of the span
    /// cannot come back.
    #[test]
    fn a_moved_reserve_settles_where_it_was_made_again() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.keep_reserve(key);
        let top = standing_arena(&pool, key);
        pool.set_arena_rank(key, top, 487);
        assert_eq!(pool.next_tombstone(key, Some(57)), Some(top));
        pool.recycle_arena_index(top);

        let remade = standing_arena(&pool, key);
        pool.set_arena_rank(key, remade, 57);
        assert_eq!(
            pool.next_tombstone(key, Some(487)),
            None,
            "the only hole is the region it left, above it: it stays"
        );
        assert_eq!(
            pool.next_tombstone(key, None),
            None,
            "nothing free at all: it stays"
        );
        assert_eq!(pool.pool_arena_load(key)[0].0, remade);
    }

    /// **A class with no reserve is swept from the top of the span down.**
    #[test]
    fn an_unreserved_class_is_swept_from_the_highest_place() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let (high, low) = reversed_pair(&pool, key);
        assert_eq!(pool.next_tombstone(key, None), Some(high));
        assert_eq!(pool.next_tombstone(key, None), Some(low));
    }

    /// **An empty arena of a reserve class goes only while the class keeps a
    /// whole arena free without it**: one slot live in the other arena leaves
    /// one slot short, so the empty arena is the reserve and stays.
    #[test]
    fn an_empty_arena_stays_while_the_class_would_fall_short_without_it() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.keep_reserve(key);
        let _first = standing_arena(&pool, key);
        let held = pool.allocate_for(key).expect("a slot in the first arena");
        let _second = standing_arena(&pool, key);
        assert_eq!(pool.free_slots(key), 2 * test_arena_chunks() - 1);
        assert_eq!(
            pool.next_tombstone(key, None),
            None,
            "one slot short of a reserve without it"
        );
        drop(held);
        assert!(
            pool.next_tombstone(key, None).is_some(),
            "both empty: one may go"
        );
        assert_eq!(pool.next_tombstone(key, None), None, "and one stays");
    }

    /// **A reserve holds one arena free and no more**: five empty arenas of a
    /// reserve class sweep down to one, however much native K/V stands.
    #[test]
    fn a_reserve_sweeps_down_to_one_free_arena() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let arena = test_arena_chunks();
        pool.keep_reserve(key);
        assert_eq!(pool.reserve_target(key), Some(arena));
        for _ in 0..5 {
            standing_arena(&pool, key);
        }
        for _ in 0..4 {
            let idx = pool.next_tombstone(key, None).expect("above the reserve");
            pool.recycle_arena_index(idx);
        }
        assert_eq!(pool.next_tombstone(key, None), None, "one arena stays");
        assert_eq!(pool.free_slots(key), arena);
    }

    /// **No reserve is kept once cleared**, for a backing that no longer
    /// writes the classes its reserves were named at.
    #[test]
    fn a_cleared_reserve_is_swept_like_any_class() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.keep_reserve(key);
        let only = standing_arena(&pool, key);
        assert_eq!(pool.next_tombstone(key, None), None);
        pool.clear_reserves();
        assert!(pool.reserve_keys().is_empty());
        assert_eq!(pool.next_tombstone(key, None), Some(only));
    }
}
