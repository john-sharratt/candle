//! Shared fixtures for the pool tests.
//!
//! The standing arena is an empty arena registered and past its creation window.
//! A reversed pair has its indices running opposite to their places in the span,
//! so a test can tell an index from a place.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in super::super) fn float_key() -> ArenaKey {
        ArenaKey::new(TEST_CLASS, ArenaLocation::Gpu)
    }

    pub(in super::super) fn test_arena_chunks() -> usize {
        TEST_CLASS.chunks_per_region()
    }

    /// An arena of `key` registered and past its creation window, empty.
    pub(in super::super) fn standing_arena(pool: &ChunkGidPool, key: ArenaKey) -> usize {
        let idx = pool.register_arena(key);
        pool.finish_creation(key, idx);
        idx
    }

    /// Two standing arenas of `key` whose indices run **opposite** to their
    /// places in the span: `(high, low)`, `high` registered first (the lower
    /// index) and carved from region 9, `low` second and from region 2.
    pub(in super::super) fn reversed_pair(pool: &ChunkGidPool, key: ArenaKey) -> (usize, usize) {
        let high = standing_arena(pool, key);
        let low = standing_arena(pool, key);
        assert!(high < low, "the lower index is the higher place");
        pool.set_arena_rank(key, high, 9);
        pool.set_arena_rank(key, low, 2);
        (high, low)
    }
}
