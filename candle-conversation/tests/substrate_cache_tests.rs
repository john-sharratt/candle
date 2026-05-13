//! Integration tests for the hot-tier [`SubstrateCache`].

use candle_conversation::substrate::{PerDepthScores, Substrate};
use candle_conversation::substrate_cache::SubstrateCache;
use candle_conversation::projection::TimelineAllocator;

/// Minimum acceptable hit rate after multiple lookups against populated entries.
const MIN_HIT_RATE: f64 = 0.90;

fn make_substrate_with_cache() -> (Substrate, SubstrateCache) {
    let cache = SubstrateCache::unbounded();
    let sub = Substrate::with_cache(cache.clone());
    (sub, cache)
}

/// Insert N turns via `append_with_blocks`, then update scores (which calls
/// `with_turn_mut` internally) on each one.  All entries are in the hot cache,
/// so every `with_turn_mut` call should be a hit.
#[test]
fn hot_cache_hit_rate_above_threshold() {
    let (mut sub, cache) = make_substrate_with_cache();

    let alloc = TimelineAllocator::new();
    let timeline = alloc.next();

    const N: usize = 20;
    let mut indices = Vec::with_capacity(N);
    for i in 0..N {
        let idx = sub.append_with_blocks(timeline, 10, i as u64 * 2, i as u64 * 2 + 1);
        indices.push(idx);
    }

    // Touch each entry via set_scores — hits the hot cache.
    for &idx in &indices {
        sub.set_scores(timeline, idx, PerDepthScores::default());
    }

    let rate = cache.hit_rate();
    assert!(
        rate >= MIN_HIT_RATE,
        "hit rate {:.1}% below minimum {:.1}%",
        rate * 100.0,
        MIN_HIT_RATE * 100.0,
    );
}

/// After `purge()`, all hot entries are gone so subsequent `with_turn_mut`
/// calls must be misses, and counters reset to zero first.
#[test]
fn purge_clears_entries_and_resets_stats() {
    let (mut sub, cache) = make_substrate_with_cache();

    let alloc = TimelineAllocator::new();
    let timeline = alloc.next();

    const N: usize = 10;
    let mut indices = Vec::with_capacity(N);
    for i in 0..N {
        let idx = sub.append_with_blocks(timeline, 5, i as u64, i as u64 + 1);
        indices.push(idx);
    }

    // Warm up some hits.
    for &idx in &indices {
        sub.set_scores(timeline, idx, PerDepthScores::default());
    }
    assert!(cache.hit_count() > 0, "expected some hits before purge");

    cache.purge();

    assert_eq!(cache.hit_count(), 0, "hit_count should reset after purge");
    assert_eq!(cache.miss_count(), 0, "miss_count should reset after purge");
    assert_eq!(cache.hot_bytes(), 0, "hot_bytes should be zero after purge");

    // Now every `with_turn_mut` is a miss — entries were evicted.
    for &idx in &indices {
        sub.set_scores(timeline, idx, PerDepthScores::default());
    }

    assert_eq!(cache.hit_count(), 0, "no hits expected after purge");
    assert_eq!(
        cache.miss_count(),
        N as u64,
        "expected exactly {} misses after purge, got {}",
        N,
        cache.miss_count()
    );
}
