//! The warm tier — the RAM-resident pool of KV chunks (§10 of
//! `docs/kv_tier_migration.md`).
//!
//! **Owned by the inference engine, not the substrate.** Bytes flow
//! through the warm pool in both directions:
//!
//! - **Write path** (VRAM → warm → NVMe). A sealed turn is gathered off
//!   the GPU into per-layer `ChunkImage` grids and inserted *dirty*. The
//!   group-commit append flushes the bytes to the redo log; the entry is
//!   then marked *clean*.
//! - **Read path** (NVMe → warm → VRAM). A cold turn's chunks are read
//!   back from the redo log and inserted *clean*. The same warm→hot leg
//!   used by warm hits ([`transfer::load_to_hot`]) materialises them
//!   into VRAM. A subsequent eviction recycles the entry through the
//!   write path.
//!
//! The pool is **LRU** with a byte budget keyed by `StreamId` (one entry
//! per turn — all layers' chunks live under one key as a `Vec<Vec<…>>`
//! grid). Under pressure clean entries drop first; dirty entries must
//! flush before they can be reclaimed.
//!
//! Each [`ChunkImage`] is *realloc-able*: it carries `KvFormat`,
//! palettes, scales, `offset`, and `token_count` alongside its KV bytes,
//! so the warm→hot leg can rebuild VRAM chunks from scratch (via the
//! allocation keystone) rather than scattering into stale GIDs.
//!
//! The pool itself is pure host logic — no CUDA dependency — and is
//! fully unit-tested without a GPU.

use std::collections::HashMap;

use super::resume::TurnChunkGrid;
use super::streams::StreamId;

/// One warm-resident stream's per-layer chunk grid plus its bookkeeping.
struct WarmEntry {
    /// The turn's per-layer ordered chunk grid.
    grid: TurnChunkGrid,
    /// Cached `grid.bytes()` — the LRU budget accounting unit.
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

    /// Insert (or replace) a stream's warm chunks. `dirty` marks the
    /// chunks as not yet durable on disk (write-path inserts); pass
    /// `false` for cold-load inserts since the bytes were already
    /// recovered from a durable redo log. Replacing an existing entry
    /// frees its old bytes.
    pub fn insert(&mut self, stream: StreamId, grid: TurnChunkGrid, dirty: bool) {
        let stamp = self.tick();
        let len = grid.bytes();
        if let Some(old) = self.entries.remove(&stream) {
            self.used_bytes -= old.bytes;
        }
        self.used_bytes += len;
        self.entries.insert(
            stream,
            WarmEntry {
                grid,
                bytes: len,
                dirty,
                last_used: stamp,
            },
        );
    }

    /// Borrow a stream's warm per-layer chunk grid, bumping its recency.
    /// `None` if not warm-resident.
    pub fn get(&mut self, stream: StreamId) -> Option<&TurnChunkGrid> {
        let stamp = self.tick();
        let entry = self.entries.get_mut(&stream)?;
        entry.last_used = stamp;
        Some(&entry.grid)
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
    use crate::persistence::resume::ChunkImage;

    fn sid(n: u64) -> StreamId {
        StreamId(n)
    }

    fn one_chunk(n_bytes: usize) -> ChunkImage {
        ChunkImage {
            token_count: 32,
            payload: ChunkPayload {
                offset: 0,
                k_formats: Vec::new(),
                v_formats: Vec::new(),
                k_pal: Vec::new(),
                v_pal: Vec::new(),
                k_scale: Vec::new(),
                v_scale: Vec::new(),
                kv_bytes: vec![0u8; n_bytes],
            },
        }
    }

    /// A one-layer, one-chunk warm grid whose KV footprint is exactly
    /// `n_bytes`.
    fn one_layer(n_bytes: usize) -> TurnChunkGrid {
        TurnChunkGrid::new(vec![vec![one_chunk(n_bytes)]])
    }

    #[test]
    fn insert_get_and_accounting() {
        let mut pool = WarmPool::new(1024);
        pool.insert(sid(1), one_layer(100), true);
        pool.insert(sid(2), one_layer(200), false);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.used_bytes(), 300);
        assert_eq!(pool.get(sid(1)).unwrap().bytes(), 100);
        assert!(pool.is_dirty(sid(1)));
        assert!(!pool.is_dirty(sid(2)));
        assert!(pool.get(sid(99)).is_none());
    }

    /// Multi-layer accounting sums across layers — the typical decoder
    /// shape (n_kv_heads × n_layers chunks per turn).
    #[test]
    fn multi_layer_grid_bytes_sum_across_layers() {
        let mut pool = WarmPool::new(10_000);
        // Three layers, each carrying a 128-byte chunk.
        let three_layers = TurnChunkGrid::new((0..3).map(|_| vec![one_chunk(128)]).collect());
        pool.insert(sid(7), three_layers, false);
        assert_eq!(pool.used_bytes(), 384);
        assert_eq!(pool.get(sid(7)).unwrap().n_layers(), 3);
    }

    #[test]
    fn replacing_an_entry_frees_old_bytes() {
        let mut pool = WarmPool::new(1024);
        pool.insert(sid(1), one_layer(500), false);
        pool.insert(sid(1), one_layer(50), false);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.used_bytes(), 50);
    }

    #[test]
    fn reclaim_drops_clean_lru_first() {
        let mut pool = WarmPool::new(250);
        pool.insert(sid(1), one_layer(100), false); // oldest, clean
        pool.insert(sid(2), one_layer(100), false); // clean
        pool.insert(sid(3), one_layer(100), false); // newest, clean -> 300 > 250
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
        pool.insert(sid(1), one_layer(100), true); // dirty
        pool.insert(sid(2), one_layer(100), true); // dirty -> 200 > 150
        let dropped = pool.reclaim();
        assert!(dropped.is_empty(), "dirty entries are not reclaimable");
        assert!(pool.over_budget(), "pool stays over budget until a flush");

        // Once flushed (clean), the LRU one becomes reclaimable.
        pool.mark_clean(sid(1));
        let dropped = pool.reclaim();
        assert_eq!(dropped, vec![sid(1)]);
        assert!(!pool.over_budget());
    }

    /// The warm pool is the rendezvous point between the eviction path
    /// (VRAM → warm, dirty) and the cold-load path (NVMe → warm, clean).
    /// Both feed the same [`TurnChunkGrid`], and a subsequent `get`
    /// returns it unchanged — this is the symmetry the warm→hot leg
    /// relies on.
    #[test]
    fn evict_and_cold_load_feed_the_same_get() {
        let mut pool = WarmPool::new(10_000);
        // Eviction-style insert (dirty).
        pool.insert(sid(1), one_layer(256), true);
        assert!(pool.is_dirty(sid(1)));
        let evicted = pool.get(sid(1)).expect("warm-after-evict").clone();

        // Cold-load-style insert (clean), same stream id. The "evict
        // came back from NVMe" lifecycle: the eviction would have been
        // marked clean post-fsync; a later cold-load arrives with the
        // exact bytes durable on disk.
        pool.insert(sid(1), one_layer(256), false);
        assert!(!pool.is_dirty(sid(1)));
        let cold = pool.get(sid(1)).expect("warm-after-cold-load").clone();

        assert_eq!(evicted.n_layers(), cold.n_layers(), "layer count matches");
        assert_eq!(
            evicted.layer(0)[0].payload.kv_bytes,
            cold.layer(0)[0].payload.kv_bytes,
            "bytes are byte-identical between eviction and cold-load arrivals"
        );
    }

    #[test]
    fn remove_frees_bytes() {
        let mut pool = WarmPool::new(1024);
        pool.insert(sid(1), one_layer(100), false);
        pool.remove(sid(1));
        assert!(pool.is_empty());
        assert_eq!(pool.used_bytes(), 0);
    }
}
