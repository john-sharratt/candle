//! The warm tier — the RAM-resident pool of evicted KV (§10 of
//! `docs/kv_tier_migration.md`).
//!
//! When a sealed sequence is evicted from VRAM its chunks land here as
//! [`ChunkImage`]s, keyed by stream. The pool is **LRU** with a byte
//! budget: under pressure it drops *clean* (already-durable-on-disk)
//! entries first; a *dirty* entry — chunks not yet flushed to the redo log
//! — must be flushed and marked clean before it can be reclaimed.
//!
//! The pool stores [`ChunkImage`]s — the *realloc-able* representation: a
//! warm entry carries each chunk's `KvFormat`, palettes, scales, `offset`
//! and `token_count` alongside its KV bytes, so `transfer::load_to_hot`
//! can rebuild the sequence's VRAM chunks from scratch (via the allocation
//! keystone) rather than scattering into stale, freed GIDs. The pool
//! itself is pure host logic and is fully unit-tested without a GPU.

use std::collections::HashMap;

use super::resume::ChunkImage;
use super::streams::StreamId;

/// Total KV byte footprint of a warm entry's chunk grid.
fn images_bytes(images: &[ChunkImage]) -> usize {
    images.iter().map(|i| i.payload.kv_bytes.len()).sum()
}

/// One warm-resident stream's chunks plus its bookkeeping.
struct WarmEntry {
    images: Vec<ChunkImage>,
    /// Cached `images_bytes(&images)` — the LRU budget accounting unit.
    bytes: usize,
    /// `true` until the chunks are durably on disk; a dirty entry cannot be
    /// reclaimed without flushing first.
    dirty: bool,
    /// Monotonic recency stamp — the largest stamp is the most recent use.
    last_used: u64,
}

/// The LRU warm pool.
pub struct WarmPool {
    entries: HashMap<StreamId, WarmEntry>,
    capacity_bytes: usize,
    used_bytes: usize,
    clock: u64,
}

impl WarmPool {
    /// A warm pool with the given byte budget.
    pub fn new(capacity_bytes: usize) -> WarmPool {
        WarmPool {
            entries: HashMap::new(),
            capacity_bytes,
            used_bytes: 0,
            clock: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Insert (or replace) a stream's warm chunks. `dirty` marks chunks not
    /// yet durable on disk. Replacing an existing entry frees its old bytes.
    pub fn insert(&mut self, stream: StreamId, images: Vec<ChunkImage>, dirty: bool) {
        let stamp = self.tick();
        let len = images_bytes(&images);
        if let Some(old) = self.entries.remove(&stream) {
            self.used_bytes -= old.bytes;
        }
        self.used_bytes += len;
        self.entries.insert(
            stream,
            WarmEntry {
                images,
                bytes: len,
                dirty,
                last_used: stamp,
            },
        );
    }

    /// Borrow a stream's warm chunks, bumping its recency. `None` if not
    /// warm-resident.
    pub fn get(&mut self, stream: StreamId) -> Option<&[ChunkImage]> {
        let stamp = self.tick();
        let entry = self.entries.get_mut(&stream)?;
        entry.last_used = stamp;
        Some(&entry.images)
    }

    /// Whether a stream is warm-resident.
    pub fn contains(&self, stream: StreamId) -> bool {
        self.entries.contains_key(&stream)
    }

    /// Whether a warm-resident stream is dirty (not yet durable).
    pub fn is_dirty(&self, stream: StreamId) -> bool {
        self.entries.get(&stream).map(|e| e.dirty).unwrap_or(false)
    }

    /// Mark a stream's bytes durable — it becomes reclaimable.
    pub fn mark_clean(&mut self, stream: StreamId) {
        if let Some(e) = self.entries.get_mut(&stream) {
            e.dirty = false;
        }
    }

    /// Drop a stream from the pool, freeing its bytes.
    pub fn remove(&mut self, stream: StreamId) {
        if let Some(e) = self.entries.remove(&stream) {
            self.used_bytes -= e.bytes;
        }
    }

    /// Reclaim clean entries in LRU order until the pool is within its byte
    /// budget. Returns the streams dropped. Dirty entries are never dropped,
    /// so the pool may stay over budget if everything resident is dirty —
    /// the caller must flush dirty entries to make them reclaimable.
    pub fn reclaim(&mut self) -> Vec<StreamId> {
        let mut dropped = Vec::new();
        while self.used_bytes > self.capacity_bytes {
            let victim = self
                .entries
                .iter()
                .filter(|(_, e)| !e.dirty)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(id, _)| *id);
            match victim {
                Some(id) => {
                    self.remove(id);
                    dropped.push(id);
                }
                // Nothing clean left to drop.
                None => break,
            }
        }
        dropped
    }

    /// Number of warm-resident streams.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes held.
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// The pool's byte budget.
    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Whether the pool is over its byte budget.
    pub fn over_budget(&self) -> bool {
        self.used_bytes > self.capacity_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::record::ChunkPayload;

    fn sid(n: u64) -> StreamId {
        StreamId(n)
    }

    /// A one-chunk warm entry whose KV footprint is exactly `n_bytes`.
    fn images(n_bytes: usize) -> Vec<ChunkImage> {
        vec![ChunkImage {
            token_count: 32,
            payload: ChunkPayload {
                offset: 0,
                k_format: 0,
                v_format: 0,
                k_pal: Vec::new(),
                v_pal: Vec::new(),
                k_scale: Vec::new(),
                v_scale: Vec::new(),
                kv_bytes: vec![0u8; n_bytes],
            },
        }]
    }

    #[test]
    fn insert_get_and_accounting() {
        let mut pool = WarmPool::new(1024);
        pool.insert(sid(1), images(100), true);
        pool.insert(sid(2), images(200), false);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.used_bytes(), 300);
        assert_eq!(images_bytes(pool.get(sid(1)).unwrap()), 100);
        assert!(pool.is_dirty(sid(1)));
        assert!(!pool.is_dirty(sid(2)));
        assert!(pool.get(sid(99)).is_none());
    }

    #[test]
    fn replacing_an_entry_frees_old_bytes() {
        let mut pool = WarmPool::new(1024);
        pool.insert(sid(1), images(500), false);
        pool.insert(sid(1), images(50), false);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.used_bytes(), 50);
    }

    #[test]
    fn reclaim_drops_clean_lru_first() {
        let mut pool = WarmPool::new(250);
        pool.insert(sid(1), images(100), false); // oldest, clean
        pool.insert(sid(2), images(100), false); // clean
        pool.insert(sid(3), images(100), false); // newest, clean -> 300 > 250
                                                 // Touch sid(1) so sid(2) is now the LRU.
        pool.get(sid(1));
        let dropped = pool.reclaim();
        assert_eq!(dropped, vec![sid(2)]);
        assert!(!pool.over_budget());
        assert!(pool.contains(sid(1)) && pool.contains(sid(3)));
    }

    #[test]
    fn reclaim_never_drops_dirty_entries() {
        let mut pool = WarmPool::new(150);
        pool.insert(sid(1), images(100), true); // dirty
        pool.insert(sid(2), images(100), true); // dirty -> 200 > 150
        let dropped = pool.reclaim();
        assert!(dropped.is_empty(), "dirty entries are not reclaimable");
        assert!(pool.over_budget(), "pool stays over budget until a flush");

        // Once flushed (clean), the LRU one becomes reclaimable.
        pool.mark_clean(sid(1));
        let dropped = pool.reclaim();
        assert_eq!(dropped, vec![sid(1)]);
        assert!(!pool.over_budget());
    }

    #[test]
    fn remove_frees_bytes() {
        let mut pool = WarmPool::new(1024);
        pool.insert(sid(1), images(100), false);
        pool.remove(sid(1));
        assert!(pool.is_empty());
        assert_eq!(pool.used_bytes(), 0);
    }
}
