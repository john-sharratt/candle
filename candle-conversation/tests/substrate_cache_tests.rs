//! Integration tests for the hot-tier [`SubstrateCache`].

use candle_conversation::projection::{SectionId, TimelineAllocator, TurnIndex};
use candle_conversation::substrate_cache::{SubstrateCache, SubstrateKey};

fn timeline_key(n: u64, idx: u32) -> SubstrateKey {
    let alloc = TimelineAllocator::new();
    // Consume (n-1) ids then take the n-th.
    for _ in 0..n {
        let _ = alloc.next();
    }
    SubstrateKey::Turn(alloc.next(), TurnIndex(idx))
}

fn section_key(id: u32) -> SubstrateKey {
    SubstrateKey::Section(SectionId::new(id))
}

/// `mark_hot` adds VRAM bytes; `mark_cold` removes them.
#[test]
fn mark_hot_cold_tracks_bytes() {
    let cache = SubstrateCache::unbounded();
    let k1 = section_key(1);
    let k2 = section_key(2);

    assert_eq!(cache.hot_bytes(), 0);

    cache.mark_hot(k1, 1024);
    assert_eq!(cache.hot_bytes(), 1024);

    cache.mark_hot(k2, 512);
    assert_eq!(cache.hot_bytes(), 1536);

    cache.mark_cold(k1);
    assert_eq!(cache.hot_bytes(), 512);

    cache.mark_cold(k2);
    assert_eq!(cache.hot_bytes(), 0);
}

/// Re-marking an already-hot key replaces the old byte count.
#[test]
fn remark_replaces_byte_count() {
    let cache = SubstrateCache::unbounded();
    let k = section_key(10);

    cache.mark_hot(k, 200);
    assert_eq!(cache.hot_bytes(), 200);

    cache.mark_hot(k, 300);
    assert_eq!(cache.hot_bytes(), 300);
}

/// Budget eviction: inserting a large entry evicts the LRU small entry.
#[test]
fn eviction_respects_budget() {
    // Budget of 1000 bytes.
    let cache = SubstrateCache::new(1000, 0, 0.0);
    let k_small = section_key(1);
    let k_large = section_key(2);

    // Fill with a 600-byte entry (never accessed → LRU timestamp 0).
    let evicted = cache.mark_hot(k_small, 600);
    assert!(evicted.is_empty());
    assert_eq!(cache.hot_bytes(), 600);

    // Insert 600 bytes more — forces eviction of k_small.
    let evicted = cache.mark_hot(k_large, 600);
    assert!(evicted.contains(&k_small), "expected k_small to be evicted");
    // After eviction k_small (600) is gone; k_large (600) is admitted.
    assert_eq!(cache.hot_bytes(), 600);
}

/// `record_access` makes an entry the most-recently used so it is NOT
/// the first to be evicted.
#[test]
fn lru_order_respects_record_access() {
    let cache = SubstrateCache::unbounded();
    let k1 = section_key(1);
    let k2 = section_key(2);
    let k3 = section_key(3);

    cache.mark_hot(k1, 0);
    cache.mark_hot(k2, 0);
    cache.mark_hot(k3, 0);

    // Access k1 last → k1 is MRU.
    cache.record_access(k2);
    cache.record_access(k3);
    cache.record_access(k1);

    let lru = cache.lru_entries(1);
    assert_eq!(lru, vec![k2], "k2 was accessed earliest so should be LRU");
}

/// `purge` clears byte accounting and resets hit/miss counters.
#[test]
fn purge_clears_entries_and_resets_stats() {
    let cache = SubstrateCache::unbounded();
    let k = section_key(42);

    cache.mark_hot(k, 256);
    cache.record_hit();
    cache.record_hit();
    cache.record_miss();

    assert_eq!(cache.hot_bytes(), 256);
    assert_eq!(cache.hit_count(), 2);
    assert_eq!(cache.miss_count(), 1);

    cache.purge();

    assert_eq!(cache.hot_bytes(), 0, "hot_bytes should be zero after purge");
    assert_eq!(cache.hit_count(), 0, "hit_count should reset after purge");
    assert_eq!(cache.miss_count(), 0, "miss_count should reset after purge");
}

/// `clear` removes accounting but preserves `hot_budget`.
#[test]
fn clear_preserves_budget() {
    let cache = SubstrateCache::new(4096, 0, 0.0);
    cache.mark_hot(section_key(1), 100);

    let budget_before = cache.hot_budget();
    cache.clear();

    assert_eq!(cache.hot_bytes(), 0);
    assert_eq!(cache.hot_budget(), budget_before, "budget must survive clear");
}

/// `hit_rate` computes correctly from explicit record calls.
#[test]
fn hit_rate_from_explicit_records() {
    let cache = SubstrateCache::unbounded();

    // 3 hits, 1 miss → 75 %
    for _ in 0..3 { cache.record_hit(); }
    cache.record_miss();

    let rate = cache.hit_rate();
    assert!(
        (rate - 0.75).abs() < 1e-9,
        "expected 75% hit rate, got {:.2}%",
        rate * 100.0,
    );
}
