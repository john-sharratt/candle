//! Tests for gid lifetimes: clone, drop, recycling, and concurrent allocation.
//!
//! A dropped slot is popped straight back off the recycle stack, and a clone
//! keeps its slot alive until the last reference drops. A detached gid counts its
//! clones without touching a pool. The concurrency test races lock-free drops
//! against gated claims, then asserts that no slot was handed out twice and that
//! every free slot is reachable once the batch is dropped.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{float_key, test_arena_chunks};

    #[test]
    fn test_register_and_allocate() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let arena_idx = pool.register_arena(key);
        assert_eq!(arena_idx, 0);

        let gid1 = pool.allocate_for(key).unwrap();
        assert_eq!(gid1.raw(), 0);
        let gid2 = pool.allocate_for(key).unwrap();
        assert_eq!(gid2.raw(), 1);
    }

    #[test]
    fn test_gid_drop_returns_to_pool() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key);

        let gid1 = pool.allocate_for(key).unwrap();
        let gid1_raw = gid1.raw();
        drop(gid1);

        // After drop, the slot is free again — total free == capacity.
        assert_eq!(pool.free_list_len_for(key), test_arena_chunks());

        // Allocating again should reuse it — the freed slot is popped straight
        // back off the recycle stack, so the same gid comes out.
        let gid2 = pool.allocate_for(key).unwrap();
        assert_eq!(gid2.raw(), gid1_raw);
    }

    #[test]
    fn test_gid_clone_no_early_return() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key);

        let gid1 = pool.allocate_for(key).unwrap();
        let gid1_clone = gid1.clone();
        let gid1_raw = gid1.raw();
        let free_after_alloc = pool.free_list_len_for(key);

        // Drop original — clone still holds a refcount.
        drop(gid1);
        assert_eq!(pool.free_list_len_for(key), free_after_alloc);

        // Drop clone — now freed.
        drop(gid1_clone);
        assert_eq!(pool.free_list_len_for(key), free_after_alloc + 1);

        let gid2 = pool.allocate_for(key).unwrap();
        assert_eq!(gid2.raw(), gid1_raw);
    }

    #[test]
    fn test_strong_count_tracks_clones() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key);
        let gid = pool.allocate_for(key).unwrap();
        assert_eq!(gid.strong_count(), 1);
        assert!(gid.is_unique());
        assert!(!gid.is_shared());

        let c1 = gid.clone();
        assert_eq!(gid.strong_count(), 2);
        assert!(gid.is_shared());

        let c2 = c1.clone();
        assert_eq!(gid.strong_count(), 3);

        drop(c2);
        assert_eq!(gid.strong_count(), 2);
        drop(c1);
        assert_eq!(gid.strong_count(), 1);
        assert!(gid.is_unique());
    }

    #[test]
    fn test_detached_clone_semantics() {
        let gid = ChunkGid::detached(-1);
        assert_eq!(gid.strong_count(), 1);
        assert!(gid.is_unique());
        assert!(gid.is_empty());
        // Detached gids carry their own per-gid refcount so cloning
        // produces a shared view, same as pooled gids.
        let c = gid.clone();
        assert_eq!(c.strong_count(), 2);
        assert!(c.is_shared());
        drop(c);
        assert_eq!(gid.strong_count(), 1);
        assert!(gid.is_unique());
        // Detached gids never touch a production pool on drop.
        drop(gid);
    }

    #[test]
    fn test_arena_register_grows_table() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let a = pool.register_arena(key);
        let b = pool.register_arena(key);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        // Each arena contributes `arena_chunks` of capacity.
        assert_eq!(pool.free_list_len_for(key), test_arena_chunks() * 2);
    }

    /// Hammer the capacity-bitmap allocator with concurrent lock-free drops
    /// racing gated allocs, filling and emptying arenas so the full↔non-full
    /// bit transitions (and their races with alloc's fill-clear) actually fire.
    /// Two invariants are asserted:
    ///   * **no double-allocation** — two threads claiming the same slot would
    ///     drive its refcount below zero on the second drop, panicking in `dec`
    ///     (`refcount underflow`), so the test would fail with that panic;
    ///   * **no permanently hidden capacity** — after every batch is dropped the
    ///     pool is fully free, so draining it must yield *exactly* the total free
    ///     count; a wrongly-cleared bit the fallback failed to recover would make
    ///     the drain come up short.
    #[test]
    fn concurrent_alloc_drop_recovers_all_capacity() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(ChunkGidPool::new());
        let key = float_key();
        pool.register_arena(key);

        let n_threads = 12;
        let batch = 400; // batches of live gids so arenas fill and empty
        let rounds = 40;
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    for _ in 0..rounds {
                        let mut held = Vec::with_capacity(batch);
                        for _ in 0..batch {
                            // Mirror `alloc_chunk_for_key`: register on exhaustion.
                            let g = loop {
                                if let Some(g) = pool.allocate_for(key) {
                                    break g;
                                }
                                pool.register_arena(key);
                            };
                            held.push(g);
                        }
                        // Drop the whole batch — lock-free `dec`s racing other
                        // threads' gated allocs (and their bit set/clear).
                        drop(held);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        // Everything dropped ⇒ the pool is fully free. Drain it and prove every
        // free slot is reachable via the bitmap fast path + the fallback scan.
        let cap = pool.free_list_len_for(key);
        assert!(cap > 0);
        let mut drained = Vec::new();
        while let Some(g) = pool.allocate_for(key) {
            drained.push(g);
        }
        assert_eq!(
            drained.len(),
            cap,
            "bitmap + fallback must reach every free slot after concurrent churn"
        );
    }
}
