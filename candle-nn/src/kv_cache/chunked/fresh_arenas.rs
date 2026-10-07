//! Arenas a compaction pass creates **ahead of demand**: kept while the pass needs them,
//! and handed back by the pass itself as soon as it knows it will not use them.
//!
//! An arena's creator always closes its creation window — at its first claim, or
//! explicitly once it has claimed (`BackingInner::claim_fresh_region_open` and
//! `ChunkGidPool::finish_creation`), or as soon as it is built
//! (`BackingInner::claim_fresh_region_then`) — so an arena nothing used is reclaimable.
//! A compaction pass is the one creator that needs an *empty* arena to survive: the low
//! arenas it provisions for each pool and the record arenas it reserves for minting are
//! made before anything lands in them, and a sweep that took one before the pass was done
//! would free an `arena_idx` the pass's plan names as a destination — or, mid-holder-sweep,
//! one that holders the sweep has not reached could come to resolve against once it is
//! re-tenanted.
//!
//! [`FreshArenas`] protects every arena the pass creates (`ChunkGidPool::protect_arena`,
//! the same exclusion the writer arenas carry) and releases the ones left empty itself,
//! at two points:
//!
//! - **After the claims** ([`FreshArenas::release_unused`]). A low arena the plan put
//!   nothing in is known to be unused once the destinations are claimed, and before the
//!   holder sweep nothing names an empty arena, so it goes back at once. That returns its
//!   hole in time for the record reservation, which provisions only while a hole exists
//!   below the frontier — without it the per-pool arenas could use every hole and leave
//!   the mints nowhere to land.
//! - **When the pass ends**, on every exit — [`FreshArenas::finish`] on success, which
//!   also reports the count, and `Drop` on a refusal or fault. Whatever the pass created
//!   and left empty is released then, and every protection it added is lifted. A pass
//!   that refuses therefore leaves no empty arena of its own standing in the holes the
//!   next pass needs.
//!
//! A released index leaves the protected set inside the release
//! (`ChunkGidPool::tombstone_if_empty`), under the lock that frees it, so it is never
//! free and protected at once.
//!
//! Admission creates arenas as the KV fills, so an empty arena handed back costs nothing
//! to recreate — and until it is handed back it is ground the wave transient tier cannot
//! stand on.

// The creators are the compaction pass and the record reservation, both CUDA-only.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use candle::Result;

use super::arena::ArenaKey;
use super::backing::BackingInner;
use super::gid_pool::ChunkGidPool;

/// What [`FreshArenas`] needs from whoever owns the arenas: the pool that protects them,
/// and a way to release one that stayed empty — pool entry and storage together.
pub(super) trait ArenaOwner {
    fn pool(&self) -> &ChunkGidPool;
    /// Release `arena_idx` if nothing is in it; `true` when it did.
    fn release_if_empty(&self, key: ArenaKey, arena_idx: usize) -> Result<bool>;
}

impl ArenaOwner for BackingInner {
    fn pool(&self) -> &ChunkGidPool {
        &self.pool
    }

    fn release_if_empty(&self, key: ArenaKey, arena_idx: usize) -> Result<bool> {
        self.release_arena_if_empty(key, arena_idx)
    }
}

/// Every arena one compaction pass created ahead of demand and still holds — see the
/// module note for when each is released.
pub(super) struct FreshArenas<'p> {
    owner: &'p dyn ArenaOwner,
    made: Vec<(ArenaKey, usize)>,
    /// Arenas created this pass, including those already released.
    created: usize,
    /// Arenas this released, for the pass's `arenas_released`.
    released: usize,
}

impl<'p> FreshArenas<'p> {
    pub(super) fn new(owner: &'p dyn ArenaOwner) -> Self {
        Self {
            owner,
            made: Vec::new(),
            created: 0,
            released: 0,
        }
    }

    /// Create one arena for `key` and hold it until this releases it.
    ///
    /// The protection goes on before the arena's creation window closes, so there is no
    /// instant at which another thread's sweep sees it both empty and unprotected.
    pub(super) fn claim(&mut self, inner: &BackingInner, key: ArenaKey) -> Result<usize> {
        inner.claim_fresh_region_then(key, |arena_idx| self.note(key, arena_idx))
    }

    fn note(&mut self, key: ArenaKey, arena_idx: usize) {
        self.owner.pool().protect_arena(arena_idx);
        self.made.push((key, arena_idx));
        self.created += 1;
    }

    /// Release every held arena nothing has landed in, and stop holding it; the count
    /// released.
    ///
    /// **Only before the holder sweep begins.** Until then no holder has been rewritten,
    /// so nothing can name an empty arena and releasing one re-tenants nothing anyone is
    /// using. The arenas that took a claim stay held.
    ///
    /// **Every arena is looked at, even after one fails.** An arena whose release errors
    /// stays held — and so is unprotected when this drops — rather than being forgotten
    /// with the rest of the list, which would leave it protected, and unreclaimable, for
    /// the life of the pool. The first error is returned once all have been tried.
    pub(super) fn release_unused(&mut self) -> Result<usize> {
        let owner = self.owner;
        let mut released = 0usize;
        let mut first_err = None;
        // A released arena leaves the protected set inside the release itself
        // (`ChunkGidPool::tombstone_if_empty`); a kept one stays protected and held.
        self.made.retain(
            |&(key, arena_idx)| match owner.release_if_empty(key, arena_idx) {
                Ok(true) => {
                    released += 1;
                    false
                }
                Ok(false) => true,
                Err(e) => {
                    first_err.get_or_insert(e);
                    true
                }
            },
        );
        self.released += released;
        match first_err {
            Some(e) => Err(e),
            None => Ok(released),
        }
    }

    /// Release everything still held that is empty, lift the protection on the rest, and
    /// answer how many arenas this released over the whole pass.
    ///
    /// What the pass calls on its way to a successful end, so the count reaches its
    /// report; an early exit gets the same release from `Drop` without the count.
    pub(super) fn finish(mut self) -> usize {
        self.release_all();
        self.released
    }

    /// Arenas created this pass, whether or not they have been released since.
    pub(super) fn created(&self) -> usize {
        self.created
    }

    fn release_all(&mut self) {
        let owner = self.owner;
        for (key, arena_idx) in self.made.drain(..) {
            match owner.release_if_empty(key, arena_idx) {
                // Out of the protected set already — see `release_unused`.
                Ok(true) => self.released += 1,
                Ok(false) => owner.pool().unprotect_arena(arena_idx),
                Err(e) => {
                    // Unprotected regardless, so the empty sweep takes the arena on its
                    // next run; this only loses the immediate hand-back.
                    tracing::error!(
                        target: "candle_nn::kv_cache::compact",
                        arena_idx,
                        ?key,
                        "could not release an arena the compaction provisioned and did not \
                         use: {e}",
                    );
                    owner.pool().unprotect_arena(arena_idx);
                }
            }
        }
    }
}

impl Drop for FreshArenas<'_> {
    fn drop(&mut self) {
        self.release_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::arena_table::ArenaLocation;
    use crate::kv_cache::chunked::size_class::SizeClass;
    use std::cell::RefCell;

    /// The pool alone, recording what it released — the pool half of what
    /// `BackingInner` does, which is all these properties depend on.
    struct PoolOwner {
        pool: ChunkGidPool,
        released: RefCell<Vec<usize>>,
    }

    impl PoolOwner {
        fn new() -> Self {
            Self {
                pool: ChunkGidPool::new(),
                released: RefCell::new(Vec::new()),
            }
        }

        /// What `claim_fresh_region_then` leaves behind before `note`: registered,
        /// storage built, nothing claimed.
        fn fresh(&self, key: ArenaKey) -> usize {
            let idx = self.pool.register_arena(key);
            self.pool.finish_creation(key, idx);
            idx
        }

        fn released(&self) -> Vec<usize> {
            let mut r = self.released.borrow().clone();
            r.sort_unstable();
            r
        }
    }

    impl ArenaOwner for PoolOwner {
        fn pool(&self) -> &ChunkGidPool {
            &self.pool
        }

        fn release_if_empty(&self, key: ArenaKey, arena_idx: usize) -> Result<bool> {
            let done = self.pool.tombstone_if_empty(key, arena_idx);
            if done {
                // There is no storage here to release; the index goes straight back,
                // as it does after `BackingInner`'s storage release.
                self.pool.recycle_arena_index(arena_idx);
                self.released.borrow_mut().push(arena_idx);
            }
            Ok(done)
        }
    }

    fn band_key() -> ArenaKey {
        ArenaKey::new(SizeClass::at(0), ArenaLocation::Gpu)
    }

    fn record_key() -> ArenaKey {
        ArenaKey::for_record_stride(ArenaLocation::Gpu, 1536)
    }

    /// Without a holder, a freshly created empty arena is reclaimable at once — the
    /// property the allocation paths rely on for an arena nothing used.
    #[test]
    fn a_fresh_arena_nobody_holds_is_reclaimable_at_once() {
        let owner = PoolOwner::new();
        let key = band_key();
        let idx = owner.fresh(key);
        assert_eq!(owner.pool.next_tombstone(key), Some(idx));
    }

    /// While the pass holds an arena no sweep can take it, and when the pass ends the
    /// pass itself hands it back.
    #[test]
    fn an_unused_arena_is_held_during_the_pass_and_released_at_its_end() {
        let owner = PoolOwner::new();
        let key = record_key();
        let idx = owner.fresh(key);
        let mut fresh = FreshArenas::new(&owner);
        fresh.note(key, idx);
        assert_eq!(fresh.created(), 1);

        assert_eq!(
            owner.pool.next_tombstone(key),
            None,
            "while the pass holds it, a provisioned arena must not be released — its \
             index could be re-tenanted under a plan or holders that still name it"
        );

        drop(fresh);
        assert_eq!(owner.released(), vec![idx], "the pass released it itself");
    }

    /// After the claims, an arena nothing landed in goes back at once and one that took a
    /// claim stays held — and stays held until the pass ends, even once empty again.
    #[test]
    fn release_unused_returns_only_what_took_no_claim() {
        let owner = PoolOwner::new();
        let key = band_key();
        let used = owner.fresh(key);
        let unused = owner.fresh(key);
        let mut fresh = FreshArenas::new(&owner);
        fresh.note(key, used);
        fresh.note(key, unused);

        let gid = owner
            .pool
            .allocate_from_arena(key, used)
            .expect("claim into the used arena");
        assert_eq!(fresh.release_unused().unwrap(), 1);
        assert_eq!(owner.released(), vec![unused]);
        assert_eq!(
            fresh.created(),
            2,
            "created counts what was made, not what is held"
        );

        drop(gid);
        assert_eq!(
            owner.pool.next_tombstone(key),
            None,
            "an arena the pass still holds is not released by a sweep even once empty"
        );
        drop(fresh);
        assert_eq!(owner.released(), vec![used, unused]);
    }

    /// An arena that took relocations is still live after the pass: the pass's own
    /// release at the end must not take an occupied arena.
    #[test]
    fn an_arena_that_received_chunks_outlives_the_pass() {
        let owner = PoolOwner::new();
        let key = band_key();
        let idx = owner.fresh(key);
        let mut fresh = FreshArenas::new(&owner);
        fresh.note(key, idx);

        let gid = owner
            .pool
            .allocate_from_arena(key, idx)
            .expect("a fresh arena serves a claim");
        drop(fresh);
        assert!(owner.released().is_empty(), "an occupied arena is kept");
        assert_eq!(owner.pool.next_tombstone(key), None);

        drop(gid);
        assert_eq!(
            owner.pool.next_tombstone(key),
            Some(idx),
            "and is an ordinary arena once the pass has gone, released with its last chunk"
        );
    }

    /// Lifting the pass's protection must not lift a protection someone else holds —
    /// the writer arenas are protected for the backing's whole life.
    #[test]
    fn the_pass_lifts_only_its_own_protection() {
        let owner = PoolOwner::new();
        let key = band_key();
        let writer = owner.fresh(key);
        owner.pool.protect_arena(writer);
        let provisioned = owner.fresh(key);
        drop({
            let mut fresh = FreshArenas::new(&owner);
            fresh.note(key, provisioned);
            fresh
        });
        assert_eq!(owner.released(), vec![provisioned]);
        assert_eq!(
            owner.pool.next_tombstone(key),
            None,
            "the writer arena stays protected after the pass"
        );
    }
}
