//! Tests for where a claim lands: the lowest placed arena, the runs, and the
//! floors that skip exhausted tails.
//!
//! A slot goes to the arena lowest in the span whatever its index, so a freed
//! slot in a low arena serves the next claim first. A run claimed by index from
//! a fresh arena succeeds even when a rival has used most of its tail, and a
//! short tail moves only the floor of its own run length. A recycled index starts
//! unplaced and is preferred again once it registers below the walk.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{
        float_key, reversed_pair, standing_arena, test_arena_chunks,
    };

    /// **A slot goes to the arena lowest in the span, whatever its index** —
    /// a single claim, a bulk claim and a run alike. Filling by index put new
    /// chunks in the high arenas compaction had just emptied, and the two moved
    /// the same chunks back and forth every few seconds.
    #[test]
    fn a_claim_takes_the_lowest_placed_arena_whatever_its_index() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let (high, low) = reversed_pair(&pool, key);

        let one = pool.allocate_for(key).expect("a slot");
        assert_eq!(one.arena_idx(), low, "single claim");
        let bulk = pool.allocate_n_for(key, 3);
        assert!(bulk.iter().all(|g| g.arena_idx() == low), "bulk claim");
        let run = pool.allocate_run_for(key, 2).expect("a run");
        assert!(run.iter().all(|g| g.arena_idx() == low), "run claim");
        assert!(pool.run_would_fit(key, 2));

        // Fill the low arena: only then does the high one serve.
        let cap = test_arena_chunks();
        let rest = pool.allocate_n_for(key, cap - 6);
        assert!(rest.iter().all(|g| g.arena_idx() == low));
        assert_eq!(pool.allocate_for(key).unwrap().arena_idx(), high);
    }

    /// **A freed slot in a full low arena brings it back first**: the drop sets
    /// the arena's capacity at its place, not at its index.
    #[test]
    fn a_drop_in_the_low_arena_serves_the_next_claim() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let (high, low) = reversed_pair(&pool, key);
        let cap = test_arena_chunks();
        let mut filled = pool.allocate_n_for(key, cap);
        assert!(filled.iter().all(|g| g.arena_idx() == low));
        let in_high = pool.allocate_for(key).unwrap();
        assert_eq!(in_high.arena_idx(), high);
        filled.pop();
        assert_eq!(pool.allocate_for(key).unwrap().arena_idx(), low);
    }

    /// **A placed arena comes before every unplaced one**, and a recycled index
    /// starts unplaced: the rank went with the table its tombstone dropped, so
    /// the index does not carry its old region's place onto a new one.
    #[test]
    fn a_recycled_index_does_not_keep_its_old_place() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let first = standing_arena(&pool, key);
        pool.set_arena_rank(key, first, 0);
        assert_eq!(pool.next_tombstone(key, None), Some(first));
        pool.recycle_arena_index(first);
        let reused = standing_arena(&pool, key);
        assert_eq!(reused, first, "the index is recycled");
        let placed = standing_arena(&pool, key);
        pool.set_arena_rank(key, placed, 7);
        assert_eq!(
            pool.allocate_for(key).unwrap().arena_idx(),
            placed,
            "the placed arena serves before the unplaced recycled one"
        );
    }

    /// The fix for the "fresh arena cannot fit palette run" race: a run claimed
    /// BY INDEX from a just-registered arena must succeed even when the global
    /// walk would have been raced, and must fail cleanly once that arena's tail
    /// is consumed (the caller then registers another).
    #[test]
    fn targeted_run_claim_hits_the_registered_arena() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();
        let idx = pool.register_arena(key);

        // Simulate the race: a rival's global walk consumes most of the fresh
        // arena's tail before our targeted claim.
        let rival = pool
            .allocate_run_for(key, cap - 2)
            .expect("rival run fits the fresh arena");
        assert_eq!(rival.len(), cap - 2);

        // The global walk can no longer fit 3 — but the targeted claim reports
        // that the SPECIFIC arena is exhausted (None), not a phantom "no arena
        // anywhere", so the caller knows to register another…
        assert!(pool.allocate_run_for(key, 3).is_none());
        assert!(pool.allocate_run_for_in(key, idx, 3).is_none());

        // …and a run that still fits the tail lands in exactly that arena.
        let run = pool
            .allocate_run_for_in(key, idx, 2)
            .expect("2 slots remain at the high-water tail");
        assert_eq!(run.len(), 2);
        assert!(run.iter().all(|g| g.arena_idx() == idx));

        // A fresh registration + targeted claim succeeds for the full length —
        // the loop the allocator runs.
        let idx2 = pool.register_arena(key);
        let run2 = pool
            .allocate_run_for_in(key, idx2, cap)
            .expect("fresh arena serves a full-capacity run");
        assert_eq!(run2.len(), cap);
        assert!(run2.iter().all(|g| g.arena_idx() == idx2));

        // Unknown arena index: None, never a panic.
        assert!(pool.allocate_run_for_in(key, 9999, 1).is_none());
    }

    /// Runs go to the lowest arena that still has a tail, and the arenas whose
    /// tails are gone are passed over — the placement the walk always gave, now
    /// without visiting every exhausted arena on every claim.
    #[test]
    fn a_run_lands_in_the_lowest_arena_with_a_tail() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();
        let a = pool.register_arena(key);
        let b = pool.register_arena(key);
        let c = pool.register_arena(key);

        // Arena `a` loses its whole tail; `b` all but two slots.
        let fill_a = pool.allocate_run_for(key, cap).unwrap();
        assert!(fill_a.iter().all(|g| g.arena_idx() == a));
        let fill_b = pool.allocate_run_for(key, cap - 2).unwrap();
        assert!(fill_b.iter().all(|g| g.arena_idx() == b));

        // The two that remain in `b` are taken before `c` is touched.
        let r1 = pool.allocate_run_for(key, 2).unwrap();
        assert!(r1.iter().all(|g| g.arena_idx() == b));
        let r2 = pool.allocate_run_for(key, 2).unwrap();
        assert!(r2.iter().all(|g| g.arena_idx() == c));

        // A new arena registered above does not displace `c`.
        pool.register_arena(key);
        let r3 = pool.allocate_run_for(key, 2).unwrap();
        assert!(r3.iter().all(|g| g.arena_idx() == c));
    }

    /// **A tail too short for a run moves that run length's floor past it, and
    /// only that length's.** Two arenas are left one slot short of empty; a run
    /// of two lands in the third and the length-2 walk starts there from now on,
    /// while a single slot still finds the lowest arena with one left. With one
    /// floor kept for "no tail at all", the one-slot tails held it at the first
    /// arena and every run claim walked the whole pool.
    #[test]
    fn a_short_tail_moves_only_its_run_lengths_floor() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();
        let a = pool.register_arena(key);
        let b = pool.register_arena(key);
        let c = pool.register_arena(key);

        let fill_a = pool.allocate_run_for(key, cap - 1).unwrap();
        assert!(fill_a.iter().all(|g| g.arena_idx() == a));
        let fill_b = pool.allocate_run_for(key, cap - 1).unwrap();
        assert!(fill_b.iter().all(|g| g.arena_idx() == b));

        // A floor is a place: these arenas carry no region, so theirs follow
        // their indices after every ranked place.
        let pair = pool.allocate_run_for(key, 2).unwrap();
        assert!(pair.iter().all(|g| g.arena_idx() == c));
        assert_eq!(pool.run_floor_for(key, 2), unranked_place(c));
        assert_eq!(pool.run_floor_for(key, 3), 0, "no length-3 walk ran");
        assert_eq!(pool.run_floor_for(key, 1), 0);

        let single = pool.allocate_run_for(key, 1).unwrap();
        assert_eq!(single[0].arena_idx(), a);
        let single = pool.allocate_run_for(key, 1).unwrap();
        assert_eq!(single[0].arena_idx(), b);
        assert_eq!(pool.run_floor_for(key, 1), unranked_place(b));
    }

    /// An arena that registers below the point the walk has passed is a fresh
    /// arena with a whole tail, and it is preferred again — lowest-first
    /// placement is what compaction packs against, and recycled indices are how
    /// a low arena comes back.
    #[test]
    fn a_recycled_low_arena_is_preferred_again_after_the_walk_has_passed_it() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();
        let a = pool.register_arena(key);
        let b = pool.register_arena(key);

        let fill_a = pool.allocate_run_for(key, cap).unwrap();
        let in_b = pool.allocate_run_for(key, 2).unwrap();
        assert!(in_b.iter().all(|g| g.arena_idx() == b));

        // `a` empties, its storage is released, and its index is recycled by the
        // next registration.
        drop(fill_a);
        assert_eq!(pool.next_tombstone(key, None), Some(a));
        pool.recycle_arena_index(a);
        assert_eq!(pool.register_arena(key), a);

        let run = pool.allocate_run_for(key, 2).unwrap();
        assert!(
            run.iter().all(|g| g.arena_idx() == a),
            "the recycled arena below the exhausted prefix has the lowest tail"
        );
    }
}
