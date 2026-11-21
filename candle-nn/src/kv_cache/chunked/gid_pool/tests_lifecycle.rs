//! Tests for when an arena's storage may be released.
//!
//! An arena made ahead of demand is reclaimable once its creator finishes without
//! claiming, while a creator that does claim still closes its own window. A
//! tombstoned index is not reissued until its storage is gone, and an empty arena
//! is reclaimed whatever the headroom, since releasing it is a constant-time
//! return. The census names each empty arena and what keeps it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{float_key, test_arena_chunks};

    /// **An arena created ahead of demand is reclaimable once its creator is
    /// done with it.**
    ///
    /// The creation window keeps a tombstoner off an index whose creator is
    /// still working on it, and every allocate-on-demand path closes that
    /// window by claiming its first chunk — so for those, "past creation" and
    /// "has been used" are the same instant and the flag can ride along with
    /// the claim. `create_deferred_arenas` stamps a slab *before* the demand
    /// arrives and has no first claim to ride, so left to `occupy` its window
    /// never closes. An arena stuck inside its window is counted by
    /// `has_reclaimable` and refused by `try_tombstone` for the life of the
    /// process, which is the pool reporting memory it cannot hand over.
    ///
    /// Measured through the substrate before this closed: one 16 MiB region
    /// stranded per pool per persistence pass, every slot in it free.
    #[test]
    fn an_arena_made_ahead_of_demand_is_reclaimable_once_its_creator_finishes() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let idx = pool.register_arena(key);

        // Registered and never claimed from: the pool counts it as recoverable…
        assert!(
            pool.has_reclaimable(),
            "an arena with every slot free is reported reclaimable"
        );
        // …and refuses to hand it over, because its creation window is open.
        // That refusal is correct while a creator still holds the index — a
        // recycled index under a live creator is cross-context contamination,
        // not a clean fault.
        assert_eq!(
            pool.next_tombstone(key, None),
            None,
            "an arena inside its creation window must not be tombstoned"
        );

        // The creator finishes without ever claiming a chunk — exactly what
        // pre-creating a slab for a class that asked for one looks like.
        pool.finish_creation(key, idx);

        assert_eq!(
            pool.next_tombstone(key, None),
            Some(idx),
            "an empty arena whose creator has finished must be reclaimable, or \
             it pins its region until the process exits"
        );
    }

    /// The explicit close did not replace the implicit one: a creator that
    /// claims a chunk still closes its own window, which is what every
    /// allocate-on-demand path relies on.
    #[test]
    fn claiming_a_chunk_still_closes_the_creation_window() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let idx = pool.register_arena(key);

        let gid = pool.allocate_for(key).expect("fresh arena serves a claim");
        assert_eq!(gid.arena_idx(), idx);
        assert_eq!(
            pool.next_tombstone(key, None),
            None,
            "an occupied arena is not reclaimable"
        );

        drop(gid);
        assert_eq!(
            pool.next_tombstone(key, None),
            Some(idx),
            "a claimed-then-dropped arena is reclaimable with no explicit \
             finish_creation — `occupy` closes the window for its creator"
        );
    }

    /// **A tombstoned index is not reissued until its storage is gone.** The pool
    /// and the arena storage release an arena in two steps, and a registration in
    /// between used to take the index while storage still held the old arena —
    /// which `ensure_arena_exists` then adopted, old stride and all, under the new
    /// key. Compaction found the result as a copy between slots of unequal extent:
    /// arena 398 holding 4096 B slots in a pool of 8192 B ones.
    #[test]
    fn a_tombstoned_index_waits_for_its_storage_release() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let other = ArenaKey::new(SizeClass::at(1), ArenaLocation::Gpu);
        let a = pool.register_arena(key);
        pool.finish_creation(key, a);
        assert_eq!(pool.next_tombstone(key, None), Some(a));
        let b = pool.register_arena(other);
        assert_ne!(
            b, a,
            "the index was reissued while its storage could still hold the old arena"
        );
        pool.recycle_arena_index(a);
        assert_eq!(
            pool.register_arena(other),
            a,
            "and is reissued once released"
        );
    }

    /// An empty arena is always reclaimed, however full the rest of the pool is.
    ///
    /// A 10 % free-headroom guard used to hold it back to stop create/destroy
    /// churn, and needed a `force` bypass because its condition ("the pool is
    /// nearly full") was exactly the state reclaim was being asked to fix.
    /// Under the reservation an arena's storage is a region handle, so there is
    /// no churn left to guard: this is the state that used to refuse.
    #[test]
    fn an_empty_arena_is_reclaimed_even_with_no_headroom_left() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();

        // Two arenas; fill BOTH completely, then free exactly one arena's worth.
        // Live is then `cap` across 2 arenas, so releasing one leaves zero free
        // slots — the case the old guard refused.
        pool.register_arena(key);
        let a: Vec<_> = (0..cap).map(|_| pool.allocate_for(key).unwrap()).collect();
        pool.register_arena(key);
        let b: Vec<_> = (0..cap).map(|_| pool.allocate_for(key).unwrap()).collect();
        assert_eq!(a.len() + b.len(), cap * 2);

        let freed_idx = b[0].arena_idx();
        assert!(
            b.iter().all(|g| g.arena_idx() == freed_idx),
            "b filled one arena"
        );
        drop(b);

        assert_eq!(
            pool.next_tombstone(key, None),
            Some(freed_idx),
            "the empty arena is reclaimed"
        );

        drop(a);
    }

    /// **The census counts every empty arena, and names what would keep each one from
    /// the sweep**: an open creation window, a protection, both, or nothing.
    #[test]
    fn the_empty_census_names_why_each_empty_arena_is_kept() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let used = pool.register_arena(key);
        let _live = pool.allocate_from_arena(key, used).unwrap();
        let _pending = pool.register_arena(key);
        let released = pool.register_arena(key);
        pool.finish_creation(key, released);
        let guarded = pool.register_arena(key);
        pool.finish_creation(key, guarded);
        pool.protect_arena(guarded);
        let both = pool.register_arena(key);
        pool.protect_arena(both);

        assert_eq!(
            pool.empty_census(),
            EmptyCensus {
                empty: 4,
                in_creation: 2,
                protected: 2,
            }
        );
        assert_eq!(pool.next_tombstone(key, None), Some(released));
        assert_eq!(pool.next_tombstone(key, None), None, "the rest are kept");
        assert_eq!(
            pool.occupancy(),
            [KeyOccupancy {
                key,
                arenas: 4,
                live: 1,
                arena_chunks: test_arena_chunks(),
            }]
        );
    }

    #[test]
    fn can_reclaim_arena_only_when_a_whole_arena_is_recoverable() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        // One arena holding a live chunk: less than a whole arena of free space,
        // so a forced compaction can release nothing.
        pool.register_arena(key);
        let _a0 = pool.allocate_from_arena(key, 0).unwrap();
        assert!(
            !pool.can_reclaim_arena(),
            "1 arena with a live chunk: nothing whole to reclaim"
        );
        // Add a second, empty arena: a whole arena's worth of free space is now
        // recoverable (needed = ceil(1 live / arena_chunks) = 1, of 2 arenas).
        pool.register_arena(key);
        assert!(
            pool.can_reclaim_arena(),
            "2 arenas, 1 live chunk: one whole arena is reclaimable"
        );
    }
}
