//! GID (Global Identifier) pool management for chunk allocation.
//!
//! Provides strongly-typed reference-counted GID allocation with automatic
//! return-to-pool semantics on drop. GIDs are partitioned by ArenaKey
//! (format + location) so allocations never land in wrong-format arenas.

use ahash::{AHashMap, AHashSet};
use candle::DType;
use std::{
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap, VecDeque},
    fmt,
    ops::Deref,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
use strum::IntoEnumIterator;

use super::arena::ArenaKey;
use crate::kv_cache::chunked::types::{arena_chunks_for_format, arena_gid_stride};
use crate::kv_cache::{ArenaLocation, KvFormat, QuantFormat};

/// Strongly-typed chunk global identifier with automatic pooling.
///
/// When dropped, the GID is automatically returned to the pool for reuse.
/// Implements reference counting semantics â€” the last holder to drop causes
/// the ID to be returned.
#[derive(Clone, Debug)]
pub struct ChunkGid {
    inner: Arc<GidInner>,
}

impl Deref for ChunkGid {
    type Target = GidInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ChunkGid {
    /// Create a detached Gid not backed by any live allocation pool.
    ///
    /// Used for sentinel handles (e.g., GID = -1) in error paths, tests, and
    /// serialisation round-trips. Dropping this Gid is a no-op on any live pool:
    /// the ID drains into an ephemeral private pool that immediately drops.
    pub fn detached(id: i64) -> Self {
        let pool = ChunkGidPool::new();
        ChunkGid {
            inner: Arc::new(GidInner {
                pool: Arc::clone(&pool.inner),
                route_key: None,
                id,
            }),
        }
    }

    /// Number of live clones of this Gid (i.e. `Arc` strong count).
    ///
    /// Used for COW detection: a count of 1 means exclusively owned;
    /// > 1 means shared (another `ChunkRef` holds a clone).
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Is this chunk shared (COW needed on write)?
    ///
    /// True when `strong_count > 1`: this ref is the structural owner;
    /// any count above 1 means a second slot or a `SealedChunk` also
    /// holds a reference.
    pub fn is_shared(&self) -> bool {
        self.strong_count() > 1
    }

    /// Check if this is the sole owner (not shared).
    pub fn is_unique(&self) -> bool {
        self.strong_count() == 1
    }
}

impl PartialEq for ChunkGid {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for ChunkGid {}

pub struct GidInner {
    pool: Arc<GidPoolInner>,
    route_key: Option<ArenaKey>,
    id: i64,
}

impl GidInner {
    /// Extract the raw i64 value.
    pub fn raw(&self) -> i64 {
        self.id
    }

    /// Returns true when this is an empty/sentinel slot (raw id == -1).
    pub fn is_empty(&self) -> bool {
        self.id == -1
    }

    /// Arena index for this GID given the per-arena chunk capacity.
    pub fn arena_idx(&self) -> usize {
        self.id as usize / arena_gid_stride()
    }

    /// Chunk offset within its arena given the global raw-GID stride.
    pub fn chunk_idx(&self) -> usize {
        self.id as usize % arena_gid_stride()
    }

    /// Arena routing key (format + location) this GID was allocated from.
    /// Returns `None` for detached/sentinel GIDs (`raw() == -1`).
    pub fn route_key(&self) -> Option<&ArenaKey> {
        self.route_key.as_ref()
    }
}

impl Drop for GidInner {
    fn drop(&mut self) {
        let Some(key) = &self.route_key else {
            return;
        };
        let Some(pool) = self.pool.pools.get(key) else {
            return;
        };
        if pool.recycle_gid(self.id) {
            self.pool
                .may_have_reclaimable
                .store(true, Ordering::Release);
        }
    }
}

impl fmt::Debug for GidInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gid").field("id", &self.id).finish()
    }
}

impl PartialEq for GidInner {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for GidInner {}

/// Per-format arena pool: min-heap of free GIDs + per-arena bookkeeping.
#[derive(Debug)]
struct ArenaPool {
    data: Mutex<ArenaPoolData>,
    /// Live chunk count for this key. Maintained incrementally so the
    /// defrag fast-path can avoid taking any locks.
    total_live: AtomicUsize,
    /// Number of currently-registered arenas for this key.
    total_arenas: AtomicUsize,
    /// Physical chunk capacity for one arena of this specific format.
    arena_chunks: usize,
}

#[derive(Debug)]
struct ArenaPoolData {
    /// Per-arena min-heaps of free GIDs.
    per_arena_free: AHashMap<usize, BinaryHeap<Reverse<i64>>>,
    /// Arena indices that currently have at least one free GID.
    non_empty_arenas: BTreeSet<usize>,
    /// Total number of free GIDs across all arenas for this key.
    total_free: usize,
    /// Per-arena free-slot count: arena_idx â†’ # of free slots.
    /// Incremented on Drop, decremented on allocate_for().
    /// When count == ARENA_CHUNKS, arena is fully reclaimable.
    free_counts: AHashMap<usize, u32>,
}

impl ArenaPool {
    fn new(format: KvFormat) -> Self {
        // Per-format arena count is small (production peak: 5 for R16, ~3 for
        // most others). 8 covers typical workloads with one rehash headroom.
        Self {
            data: Mutex::new(ArenaPoolData {
                per_arena_free: AHashMap::with_capacity(8),
                non_empty_arenas: BTreeSet::new(),
                total_free: 0,
                free_counts: AHashMap::with_capacity(8),
            }),
            total_live: AtomicUsize::new(0),
            total_arenas: AtomicUsize::new(0),
            arena_chunks: arena_chunks_for_format(format),
        }
    }

    fn seed_arena(&self, arena_idx: usize) {
        let mut data = self.data.lock().unwrap();
        let base = (arena_idx * arena_gid_stride()) as i64;
        // Build the free list in one shot. Size is known up front
        // (`arena_chunks`), so pre-allocate the underlying Vec exactly once,
        // fill it with a tight loop, then hand it to `BinaryHeap::from` which
        // heapifies in place in O(N) via Floyd's algorithm — no growth thrash,
        // no per-element sift-up. For Q0 (524288 chunks) this turns ~half a
        // million individual heap pushes into one allocation + one linear pass.
        let n = self.arena_chunks;
        let mut buf: Vec<Reverse<i64>> = Vec::with_capacity(n);
        for i in 0..n as i64 {
            buf.push(Reverse(base + i));
        }
        let heap = BinaryHeap::from(buf);
        data.per_arena_free.insert(arena_idx, heap);
        data.non_empty_arenas.insert(arena_idx);
        data.total_free += self.arena_chunks;
        data.free_counts.insert(arena_idx, self.arena_chunks as u32);
        self.total_arenas.fetch_add(1, Ordering::Relaxed);
    }

    fn allocate_any(&self) -> Option<i64> {
        let mut data = self.data.lock().ok()?;
        let arena_idx = *data.non_empty_arenas.iter().next()?;
        let gid = {
            let heap = data.per_arena_free.get_mut(&arena_idx)?;
            let Reverse(gid) = heap.pop()?;
            gid
        };
        if data
            .per_arena_free
            .get(&arena_idx)
            .map(|h| h.is_empty())
            .unwrap_or(true)
        {
            data.non_empty_arenas.remove(&arena_idx);
        }
        data.total_free = data.total_free.saturating_sub(1);
        let count = data
            .free_counts
            .get_mut(&arena_idx)
            .expect("arena in free pool but missing from free_counts");
        *count -= 1;
        self.total_live.fetch_add(1, Ordering::Relaxed);
        Some(gid)
    }

    fn allocate_from_arena(&self, target_arena: usize) -> Option<i64> {
        let mut data = self.data.lock().ok()?;
        let gid = {
            let heap = data.per_arena_free.get_mut(&target_arena)?;
            let Reverse(gid) = heap.pop()?;
            gid
        };
        if data
            .per_arena_free
            .get(&target_arena)
            .map(|h| h.is_empty())
            .unwrap_or(true)
        {
            data.non_empty_arenas.remove(&target_arena);
        }
        data.total_free = data.total_free.saturating_sub(1);
        let count = data
            .free_counts
            .get_mut(&target_arena)
            .expect("arena in free_counts");
        *count -= 1;
        self.total_live.fetch_add(1, Ordering::Relaxed);
        Some(gid)
    }

    fn recycle_gid(&self, id: i64) -> bool {
        let mut data = self.data.lock().unwrap();
        let arena_idx = id as usize / arena_gid_stride();
        data.per_arena_free
            .entry(arena_idx)
            .or_insert_with(BinaryHeap::new)
            .push(Reverse(id));
        data.non_empty_arenas.insert(arena_idx);
        data.total_free += 1;
        let count = data.free_counts.entry(arena_idx).or_insert(0);
        *count += 1;
        self.total_live.fetch_sub(1, Ordering::Relaxed);
        *count == self.arena_chunks as u32
    }

    fn try_tombstone(&self, protected_arenas: &AHashSet<usize>) -> Option<usize> {
        let mut data = self.data.lock().ok()?;
        let arena_idx = {
            let idx = data
                .free_counts
                .iter()
                .find(|(&idx, &count)| {
                    count == self.arena_chunks as u32 && !protected_arenas.contains(&idx)
                })
                .map(|(&idx, _)| idx)?;

            let after_release = data.total_free.saturating_sub(self.arena_chunks);
            if after_release < self.arena_chunks / 10 {
                return None;
            }

            idx
        };

        data.per_arena_free.remove(&arena_idx);
        data.non_empty_arenas.remove(&arena_idx);
        data.total_free = data.total_free.saturating_sub(self.arena_chunks);
        data.free_counts.remove(&arena_idx);
        self.total_arenas.fetch_sub(1, Ordering::Relaxed);

        Some(arena_idx)
    }

    fn has_reclaimable(&self) -> bool {
        self.data
            .lock()
            .map(|data| {
                data.free_counts
                    .values()
                    .any(|&count| count == self.arena_chunks as u32)
            })
            .unwrap_or(false)
    }

    fn free_count_for_arena(&self, arena_idx: usize) -> u32 {
        self.data
            .lock()
            .ok()
            .and_then(|data| data.free_counts.get(&arena_idx).copied())
            .unwrap_or(0)
    }

    /// Allocate one GID from any arena except `exclude_arena`.
    ///
    /// Used during greedy arena eviction so that destination slots are never
    /// placed in the arena we are trying to drain.
    #[allow(dead_code)]
    fn allocate_excluding(&self, exclude_arena: usize) -> Option<i64> {
        let mut data = self.data.lock().ok()?;
        let arena_idx = data
            .non_empty_arenas
            .iter()
            .find(|&&idx| idx != exclude_arena)
            .copied()?;
        let gid = {
            let heap = data.per_arena_free.get_mut(&arena_idx)?;
            let Reverse(gid) = heap.pop()?;
            gid
        };
        if data
            .per_arena_free
            .get(&arena_idx)
            .map(|h| h.is_empty())
            .unwrap_or(true)
        {
            data.non_empty_arenas.remove(&arena_idx);
        }
        data.total_free = data.total_free.saturating_sub(1);
        let count = data
            .free_counts
            .get_mut(&arena_idx)
            .expect("arena in free pool but missing from free_counts");
        *count -= 1;
        self.total_live.fetch_add(1, Ordering::Relaxed);
        Some(gid)
    }

    /// Return the live (allocated) GIDs for `arena_idx`.
    ///
    /// Computes live = all slots in the arena minus those in the free heap.
    /// Acquires only the pool's data lock — no metadata or sequence scan needed.
    #[allow(dead_code)]
    fn live_gids_for_arena(&self, arena_idx: usize) -> Vec<i64> {
        let Ok(data) = self.data.lock() else {
            return Vec::new();
        };
        let base = (arena_idx * arena_gid_stride()) as i64;
        let free_set: AHashSet<i64> = data
            .per_arena_free
            .get(&arena_idx)
            .map(|heap| {
                let mut s = AHashSet::with_capacity(heap.len());
                s.extend(heap.iter().map(|Reverse(id)| *id));
                s
            })
            .unwrap_or_default();
        (0..self.arena_chunks as i64)
            .map(|i| base + i)
            .filter(|id| !free_set.contains(id))
            .collect()
    }

    /// Return all arenas for this pool sorted by live chunk count ascending
    /// (emptiest drain candidate first). Each entry is (arena_idx, live_count).
    #[allow(dead_code)]
    fn arenas_sorted_by_live(&self) -> Vec<(usize, usize)> {
        let Ok(data) = self.data.lock() else {
            return Vec::new();
        };
        let capacity = self.arena_chunks;
        let mut arenas: Vec<(usize, usize)> = data
            .free_counts
            .iter()
            .map(|(&idx, &free)| (idx, capacity.saturating_sub(free as usize)))
            .collect();
        arenas.sort_by_key(|&(_, live)| live);
        arenas
    }

    /// Fraction of registered arenas that could be fully reclaimed by a
    /// perfect pack-down to the minimum arena count.
    fn defragmentable_ratio(&self) -> f32 {
        let arenas = self.total_arenas.load(Ordering::Relaxed);
        if arenas == 0 {
            return 0.0;
        }
        let live = self.total_live.load(Ordering::Relaxed);
        let needed = if live == 0 {
            0
        } else {
            live.div_ceil(self.arena_chunks)
        };
        (arenas.saturating_sub(needed)) as f32 / arenas as f32
    }

    /// Rebuild the lock-free counters from the authoritative heap/free-count
    /// state so any transient drift is corrected after a full compaction pass.
    fn resync_counters(&self) {
        if let Ok(data) = self.data.lock() {
            let arenas = data.free_counts.len();
            let total_slots = arenas.saturating_mul(self.arena_chunks);
            let live = total_slots.saturating_sub(data.total_free);
            self.total_arenas.store(arenas, Ordering::Relaxed);
            self.total_live.store(live, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    fn free_len(&self) -> usize {
        self.data.lock().map(|data| data.total_free).unwrap_or(0)
    }

    #[cfg(test)]
    fn inject_free_id(&self, id: i64) {
        let mut data = self.data.lock().unwrap();
        let arena_idx = id as usize / arena_gid_stride();
        data.per_arena_free
            .entry(arena_idx)
            .or_insert_with(BinaryHeap::new)
            .push(Reverse(id));
        data.non_empty_arenas.insert(arena_idx);
        data.total_free += 1;
        *data.free_counts.entry(arena_idx).or_insert(0) += 1;
    }
}

/// Internal state of the GID pool, partitioned by ArenaKey.
struct GidPoolState {
    /// arena_idx â†’ ArenaKey registry for routing/compaction bookkeeping.
    /// None = tombstoned (index recycled) or not yet assigned.
    arena_registry: Vec<Option<ArenaKey>>,
    /// Monotonic arena index allocator (fallback when free_arenas is empty).
    next_arena_idx: usize,
    /// FIFO queue of recycled arena indices from tombstoned arenas.
    free_arenas: VecDeque<usize>,
    /// Arena indices pinned for the lifetime of this backing.
    /// Protected arenas are never tombstoned by compaction.
    protected_arenas: AHashSet<usize>,
}

/// Inner pool state, protected by its own lock.
struct GidPoolInner {
    /// Immutable key->pool table. Each pool has its own lock for short critical sections.
    pools: AHashMap<ArenaKey, ArenaPool>,
    /// Mutable metadata for arena routing/recycling.
    metadata: Mutex<GidPoolState>,
    /// Lock-free hint for reclaim detection.
    /// false => definitely no fully-free arena.
    /// true  => maybe reclaimable; verify under lock.
    may_have_reclaimable: AtomicBool,
}

fn preallocated_pool_table() -> AHashMap<ArenaKey, ArenaPool> {
    // Locations × (DType variants + QuantFormat variants). With current enums
    // this is ~58 entries; 64 leaves room without rehashing on first registration.
    let mut pools = AHashMap::with_capacity(64);

    for location in ArenaLocation::iter() {
        for dtype in DType::iter() {
            pools.insert(
                ArenaKey::new(KvFormat::Float(dtype), location),
                ArenaPool::new(KvFormat::Float(dtype)),
            );
        }
        for qf in QuantFormat::iter() {
            pools.insert(
                ArenaKey::new(KvFormat::Quantized(qf), location),
                ArenaPool::new(KvFormat::Quantized(qf)),
            );
        }
    }

    pools
}

/// Pool for allocating and recycling GIDs, partitioned by ArenaKey.
///
/// Each `ChunkGidPool` manages per-format arena pools with min-heap free lists.
/// GIDs are returned to the correct pool on Drop via the arena_registry.
pub struct ChunkGidPool {
    inner: Arc<GidPoolInner>,
}

impl ChunkGidPool {
    /// Create a new GID pool with no registered arenas.
    pub fn new() -> Self {
        // arena_registry/free_arenas track total live + recycled arena indices
        // across all formats — production peak is ~28, so 64 covers headroom.
        // protected_arenas holds the warm baseline + quant candidates; bounded
        // by the size of the warm set (~10-20 entries).
        Self {
            inner: Arc::new(GidPoolInner {
                pools: preallocated_pool_table(),
                metadata: Mutex::new(GidPoolState {
                    arena_registry: Vec::with_capacity(64),
                    next_arena_idx: 0,
                    free_arenas: VecDeque::with_capacity(16),
                    protected_arenas: AHashSet::with_capacity(32),
                }),
                may_have_reclaimable: AtomicBool::new(false),
            }),
        }
    }

    /// Mark an arena index as protected from compaction tombstoning.
    ///
    /// Protected arenas remain allocated until the owning backing is dropped.
    pub fn protect_arena(&self, arena_idx: usize) {
        let mut state = self.inner.metadata.lock().unwrap();
        state.protected_arenas.insert(arena_idx);
    }

    /// Register a new arena with the pool.
    ///
    /// Recycles a tombstoned arena index (FIFO) or assigns the next fresh index.
    /// Bulk-inserts `ARENA_CHUNKS` GIDs into the format's min-heap and returns
    /// the assigned arena index. The caller uses this index to position the
    /// arena in storage.
    pub fn register_arena(&self, key: ArenaKey) -> usize {
        let arena_idx = {
            let mut state = self.inner.metadata.lock().unwrap();
            // Recycle a tombstoned index or assign fresh
            let arena_idx = state.free_arenas.pop_front().unwrap_or_else(|| {
                let idx = state.next_arena_idx;
                state.next_arena_idx += 1;
                idx
            });
            // Grow registry if needed
            if arena_idx >= state.arena_registry.len() {
                state.arena_registry.resize(arena_idx + 1, None);
            }
            state.arena_registry[arena_idx] = Some(key.clone());
            arena_idx
        };

        let pool = self
            .inner
            .pools
            .get(&key)
            .expect("register_arena: missing preallocated pool for key");
        pool.seed_arena(arena_idx);
        self.inner
            .may_have_reclaimable
            .store(true, Ordering::Release);

        arena_idx
    }

    /// Allocate a GID for the given format.
    ///
    /// Pops the lowest GID from the per-format min-heap. Returns None if no
    /// capacity â€” caller should create a new arena via register_arena(), then retry.
    pub fn allocate_for(&self, key: ArenaKey) -> Option<ChunkGid> {
        let pool = self.inner.pools.get(&key)?;
        let gid = pool.allocate_any()?;

        Some(ChunkGid {
            inner: Arc::new(GidInner {
                pool: Arc::clone(&self.inner),
                route_key: Some(key),
                id: gid,
            }),
        })
    }

    /// Allocate a GID from a specific arena (for consolidation that must target
    /// a particular destination arena). Returns None if that arena is full or
    /// not registered.
    pub fn allocate_from_arena(&self, key: ArenaKey, target_arena: usize) -> Option<ChunkGid> {
        let pool = self.inner.pools.get(&key)?;
        let gid = pool.allocate_from_arena(target_arena)?;

        Some(ChunkGid {
            inner: Arc::new(GidInner {
                pool: Arc::clone(&self.inner),
                route_key: Some(key),
                id: gid,
            }),
        })
    }

    /// Allocate a GID for `key` from any arena **except** `exclude_arena`.
    ///
    /// Used during greedy arena eviction so destination slots never land in
    /// the arena being drained. Returns `None` when no other arena has free
    /// capacity — the caller should skip this eviction attempt gracefully.
    #[cfg(feature = "cuda")]
    pub fn allocate_for_excluding(&self, key: ArenaKey, exclude_arena: usize) -> Option<ChunkGid> {
        let pool = self.inner.pools.get(&key)?;
        let gid = pool.allocate_excluding(exclude_arena)?;
        Some(ChunkGid {
            inner: Arc::new(GidInner {
                pool: Arc::clone(&self.inner),
                route_key: Some(key),
                id: gid,
            }),
        })
    }

    /// Return the live GIDs for `arena_idx` (all slots minus free slots).
    ///
    /// Derived purely from pool state — no sequence scan needed.
    #[cfg(feature = "cuda")]
    pub fn live_gids_for_arena(&self, arena_idx: usize) -> Vec<i64> {
        let key = {
            let Ok(state) = self.inner.metadata.lock() else {
                return Vec::new();
            };
            match state.arena_registry.get(arena_idx) {
                Some(Some(k)) => k.clone(),
                _ => return Vec::new(),
            }
        };
        self.inner
            .pools
            .get(&key)
            .map(|pool| pool.live_gids_for_arena(arena_idx))
            .unwrap_or_default()
    }

    /// Return arenas for `key` sorted by live chunk count ascending.
    ///
    /// Each entry is `(arena_idx, live_count)`. The emptiest arena (best drain
    /// candidate) comes first. Acquires only the pool's data lock.
    #[cfg(feature = "cuda")]
    pub fn arenas_sorted_by_live_for_key(&self, key: &ArenaKey) -> Vec<(usize, usize)> {
        self.inner
            .pools
            .get(key)
            .map(|pool| pool.arenas_sorted_by_live())
            .unwrap_or_default()
    }

    /// Register an externally-created arena at a specific arena index.
    ///
    /// This is used when chunk_ops creates a quant or float arena at a known
    /// index (e.g., after `ArenaStorageState::arenas.len()` grows). Unlike
    /// `register_arena` which assigns the next free index, this registers at
    /// exactly the given index â€” necessary when ArenaStorageState and GidPool
    /// must agree on the arena index.
    ///
    /// Panics if the index is already registered.
    pub fn register_arena_at(&self, arena_idx: usize, key: ArenaKey) {
        {
            let mut state = self.inner.metadata.lock().unwrap();

            if arena_idx >= state.arena_registry.len() {
                state.arena_registry.resize(arena_idx + 1, None);
            }
            assert!(
                state.arena_registry[arena_idx].is_none(),
                "register_arena_at: arena index {arena_idx} already registered"
            );
            state.arena_registry[arena_idx] = Some(key.clone());
        }

        let pool = self
            .inner
            .pools
            .get(&key)
            .expect("register_arena_at: missing preallocated pool for key");
        pool.seed_arena(arena_idx);
        self.inner
            .may_have_reclaimable
            .store(true, Ordering::Release);
    }

    /// Convenience: allocate a GID using a default test key.
    ///
    /// Auto-registers a new arena if no capacity remains. Intended for tests
    /// and test fixtures that don't care about format partitioning.
    pub fn allocate(&self) -> ChunkGid {
        let key = ArenaKey::gpu_float(candle::DType::BF16);
        if let Some(gid) = self.allocate_for(key.clone()) {
            return gid;
        }
        // No capacity â€” register a new arena and retry
        self.register_arena(key.clone());
        self.allocate_for(key)
            .expect("just registered arena, must have capacity")
    }

    /// Find a fully-free arena of this format and release it.
    ///
    /// Returns Some(arena_idx) if an arena has all its slots returned
    /// AND the free list would retain â‰¥ 10% of ARENA_CHUNKS capacity after
    /// releasing. Drains the arena's GIDs from the heap, clears the arena from
    /// the registry, and recycles the index via FIFO.
    /// Returns None if no arena qualifies.
    pub fn next_tombstone(&self, key: ArenaKey) -> Option<usize> {
        let mut state = self.inner.metadata.lock().unwrap();
        let pool = self.inner.pools.get(&key)?;
        let arena_idx = pool.try_tombstone(&state.protected_arenas)?;
        state.arena_registry[arena_idx] = None;
        state.free_arenas.push_back(arena_idx);

        Some(arena_idx)
    }

    /// Remove from `free_arenas` any indices >= `threshold`.
    ///
    /// Called after `storage.truncate_arenas(threshold)` to keep the pool's
    /// recycle queue in sync with storage.  Without this, a truncated index can
    /// be handed out by `register_arena`, then `ensure_arena_exists` sees
    /// `current >= needed` (because a *different* registration already grew
    /// storage past that index) and returns early without creating the correct
    /// arena, causing a format mismatch at allocation time.
    pub fn drain_free_arenas_above(&self, threshold: usize) {
        let mut state = self.inner.metadata.lock().unwrap();
        state.free_arenas.retain(|&idx| idx < threshold);
    }

    /// Check whether any arena is fully free (all ARENA_CHUNKS GIDs returned).
    ///
    /// Fast path is lock-free: if the atomic hint is false, we can return false
    /// immediately. When the hint is true, verify under lock and clear if stale.
    pub fn has_reclaimable(&self) -> bool {
        if !self.inner.may_have_reclaimable.load(Ordering::Acquire) {
            return false;
        }

        let any = self.inner.pools.values().any(|pool| pool.has_reclaimable());
        if !any {
            self.inner
                .may_have_reclaimable
                .store(false, Ordering::Release);
        }
        any
    }

    /// Return the set of format keys currently registered in the pool.
    pub fn format_keys(&self) -> Vec<ArenaKey> {
        let state = self.inner.metadata.lock().unwrap();
        let mut out: AHashSet<ArenaKey> = AHashSet::new();
        for key in state.arena_registry.iter().flatten() {
            out.insert(key.clone());
        }
        out.into_iter().collect()
    }

    /// Lock-free hint: return true if any key can free at least `threshold`
    /// of its currently registered arenas via perfect defragmentation.
    pub(crate) fn needs_defragmentation(&self, threshold: f32) -> bool {
        if threshold <= 0.0 {
            return self
                .inner
                .pools
                .values()
                .any(|pool| pool.total_arenas.load(Ordering::Relaxed) > 0);
        }
        self.inner
            .pools
            .values()
            .any(|pool| pool.defragmentable_ratio() > threshold)
    }

    /// Lock-free defragmentable ratio for a specific key.
    #[allow(dead_code)]
    pub(crate) fn defragmentable_ratio_for(&self, key: &ArenaKey) -> f32 {
        self.inner
            .pools
            .get(key)
            .map(|pool| pool.defragmentable_ratio())
            .unwrap_or(0.0)
    }

    /// Get the maximum GID currently in circulation (for compaction bound checks).
    pub fn max_gid(&self) -> Option<i64> {
        let state = self.inner.metadata.lock().ok()?;
        let mut max: i64 = -1;
        // Check all registered arenas (live or with free GIDs)
        for (idx, entry) in state.arena_registry.iter().enumerate() {
            if entry.is_some() {
                let arena_top = ((idx + 1) * arena_gid_stride()) as i64 - 1;
                if arena_top > max {
                    max = arena_top;
                }
            }
        }
        if max >= 0 {
            Some(max)
        } else {
            None
        }
    }

    /// Rebuild the fast-path counters for all format pools from their
    /// authoritative free-list state.
    pub(crate) fn resync_counters(&self) {
        for pool in self.inner.pools.values() {
            pool.resync_counters();
        }
    }

    /// Get the total number of free GIDs across all formats (for testing).
    #[cfg(test)]
    pub(crate) fn total_free(&self) -> usize {
        self.inner.pools.values().map(|pool| pool.free_len()).sum()
    }

    /// Get the number of free GIDs for a specific arena index.
    /// Returns 0 if the arena is not registered or has no free GIDs.
    pub(crate) fn arena_free_count(&self, arena_idx: usize) -> u32 {
        let state = self.inner.metadata.lock().unwrap();
        let key = match state.arena_registry.get(arena_idx) {
            Some(Some(k)) => k.clone(),
            _ => return 0,
        };
        self.inner
            .pools
            .get(&key)
            .map(|p| p.free_count_for_arena(arena_idx))
            .unwrap_or(0)
    }

    /// Get the free list length for a specific format (for testing).
    #[cfg(test)]
    pub(crate) fn free_list_len_for(&self, key: ArenaKey) -> usize {
        self.inner
            .pools
            .get(&key)
            .map(|p| p.free_len())
            .unwrap_or(0)
    }

    /// Push a raw id directly into a format's free list (test / injection only).
    #[cfg(test)]
    pub(crate) fn free(&self, key: ArenaKey, id: i64) {
        if let Some(pool) = self.inner.pools.get(&key) {
            pool.inject_free_id(id);
        }
    }

    /// Number of registered arenas (for testing).
    #[cfg(test)]
    pub(crate) fn arena_count(&self) -> usize {
        let state = self.inner.metadata.lock().unwrap();
        state.arena_registry.iter().filter(|e| e.is_some()).count()
    }
}

impl Default for ChunkGidPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ChunkGidPool {
    /// Clone the pool â€” shares the same internal state.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl fmt::Debug for ChunkGidPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GidPool").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;

    fn float_key() -> ArenaKey {
        ArenaKey::gpu_float(DType::BF16)
    }

    fn test_arena_chunks() -> usize {
        arena_chunks_for_format(KvFormat::Float(DType::BF16))
    }

    fn test_gid_stride() -> usize {
        arena_gid_stride()
    }

    #[test]
    fn test_register_and_allocate() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let arena_idx = pool.register_arena(key.clone());
        assert_eq!(arena_idx, 0);

        // First allocation should return GID 0 (lowest in arena 0)
        let gid1 = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid1.raw(), 0);
        let gid2 = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid2.raw(), 1);
    }

    #[test]
    fn test_gid_drop_returns_to_pool() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone());

        let gid1 = pool.allocate_for(key.clone()).unwrap();
        let gid1_raw = gid1.raw();
        drop(gid1);

        // After drop, the GID should be back in the free list
        assert_eq!(pool.free_list_len_for(key.clone()), test_arena_chunks()); // all slots free again

        // Allocating again should reuse it (lowest GID)
        let gid2 = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid2.raw(), gid1_raw);
    }

    #[test]
    fn test_gid_clone_no_early_return() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone());

        let gid1 = pool.allocate_for(key.clone()).unwrap();
        let gid1_clone = gid1.clone();
        let gid1_raw = gid1.raw();
        let free_after_alloc = pool.free_list_len_for(key.clone());

        // Drop original - clone still holds Arc<GidInner>
        drop(gid1);
        assert_eq!(pool.free_list_len_for(key.clone()), free_after_alloc); // not freed yet

        // Drop clone - now freed
        drop(gid1_clone);
        assert_eq!(pool.free_list_len_for(key.clone()), free_after_alloc + 1);

        let gid2 = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid2.raw(), gid1_raw);
    }

    #[test]
    fn test_defragmentation_ratio_only_triggers_when_arenas_can_be_freed() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone());
        pool.register_arena(key.clone());

        // Force sparse live chunks across two arenas.
        let _a0 = pool.allocate_from_arena(key.clone(), 0).unwrap();
        let _a1 = pool.allocate_from_arena(key.clone(), 1).unwrap();

        let ratio = pool.defragmentable_ratio_for(&key);
        assert!(ratio > 0.4, "expected meaningful defrag ratio, got {ratio}");
        assert!(pool.needs_defragmentation(0.2));

        // A single non-empty arena should not be considered defragmentable.
        let pool2 = ChunkGidPool::new();
        pool2.register_arena(key.clone());
        let _only = pool2.allocate_for(key.clone()).unwrap();
        assert_eq!(pool2.defragmentable_ratio_for(&key), 0.0);
        assert!(!pool2.needs_defragmentation(0.2));
    }

    #[test]
    fn test_allocate_returns_lowest_gid() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone());

        let gid0 = pool.allocate_for(key.clone()).unwrap();
        let gid1 = pool.allocate_for(key.clone()).unwrap();
        let gid2 = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid0.raw(), 0);
        assert_eq!(gid1.raw(), 1);
        assert_eq!(gid2.raw(), 2);

        // Drop gid0 and gid2, keep gid1
        drop(gid0);
        drop(gid2);

        // Should get 0 first (lowest), then 2
        let gid3 = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid3.raw(), 0);
        let gid4 = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid4.raw(), 2);
    }

    #[test]
    fn test_pool_clone_shares_state() {
        let pool1 = ChunkGidPool::new();
        let pool2 = pool1.clone();
        let key = float_key();

        pool1.register_arena(key.clone());

        let gid1 = pool1.allocate_for(key.clone()).unwrap();
        assert_eq!(gid1.raw(), 0);

        // pool2 sees pool1's state
        let gid2 = pool2.allocate_for(key.clone()).unwrap();
        assert_eq!(gid2.raw(), 1);

        drop(gid1);
        assert_eq!(
            pool2.free_list_len_for(key.clone()),
            test_arena_chunks() - 1
        ); // 512 - 2 + 1 dropped
    }

    #[test]
    fn test_multiple_arenas_same_format() {
        let pool = ChunkGidPool::new();
        let key = float_key();

        let idx0 = pool.register_arena(key.clone());
        let idx1 = pool.register_arena(key.clone());
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);

        // Should allocate from arena 0 first (lowest GIDs)
        let gid = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid.raw(), 0);
        assert_eq!(gid.arena_idx(), 0);
    }

    #[test]
    fn test_format_isolation() {
        let pool = ChunkGidPool::new();
        let float_k = ArenaKey::gpu_float(DType::BF16);
        let quant_k = ArenaKey::gpu_quant(crate::kv_cache::QuantFormat::Q8_0);

        pool.register_arena(float_k.clone());
        pool.register_arena(quant_k.clone());

        // Float allocations come from arena 0
        let fgid = pool.allocate_for(float_k.clone()).unwrap();
        assert_eq!(fgid.arena_idx(), 0);

        // Quant allocations come from arena 1
        let qgid = pool.allocate_for(quant_k.clone()).unwrap();
        assert_eq!(qgid.arena_idx(), 1);

        // No cross-contamination
        assert!(pool.allocate_for(ArenaKey::gpu_float(DType::F32)).is_none());
    }

    #[test]
    fn test_next_tombstone() {
        let pool = ChunkGidPool::new();
        let key = float_key();

        // Register 3 arenas: we need enough free capacity after releasing one
        pool.register_arena(key.clone()); // idx 0
        pool.register_arena(key.clone()); // idx 1
        pool.register_arena(key.clone()); // idx 2

        // Allocate all of arena 0, leave arenas 1 and 2 completely free
        let mut gids = Vec::new();
        for _ in 0..test_arena_chunks() {
            gids.push(pool.allocate_for(key.clone()).unwrap());
        }
        // All arena 0 GIDs allocated; arenas 1 and 2 are fully free (1024 free)

        // next_tombstone should find a fully free arena (1 or 2)
        let tombstoned = pool.next_tombstone(key.clone());
        assert!(tombstoned == Some(1) || tombstoned == Some(2));

        // That arena's GIDs should be drained; registry entry removed
        assert_eq!(pool.arena_count(), 2);
    }

    #[test]
    fn test_arena_index_recycling() {
        let pool = ChunkGidPool::new();
        let key = float_key();

        // Register 3 arenas so tombstone threshold is met
        pool.register_arena(key.clone()); // idx 0
        pool.register_arena(key.clone()); // idx 1
        pool.register_arena(key.clone()); // idx 2

        // Allocate everything from arena 0
        let mut gids = Vec::new();
        for _ in 0..test_arena_chunks() {
            gids.push(pool.allocate_for(key.clone()).unwrap());
        }

        // Tombstone a fully-free arena
        let tombstoned = pool.next_tombstone(key.clone()).unwrap();
        assert!(tombstoned == 1 || tombstoned == 2);

        // Register a new arena â€” should recycle the tombstoned index
        let recycled = pool.register_arena(key.clone());
        assert_eq!(recycled, tombstoned);
    }

    #[test]
    fn test_gid_debug_format() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone());
        let gid = pool.allocate_for(key.clone()).unwrap();

        let debug_str = format!("{:?}", gid);
        assert!(debug_str.contains("Gid"));
        assert!(debug_str.contains("id: 0"));
    }

    // ── allocate_from_arena ──────────────────────────────────────────────────

    #[test]
    fn test_allocate_from_arena_targets_specific_arena() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone()); // idx 0
        pool.register_arena(key.clone()); // idx 1
        pool.register_arena(key.clone()); // idx 2

        // Fill arena 0 and arena 1 completely so only arena 2 has free slots
        let mut consumed: Vec<ChunkGid> = Vec::new();
        for _ in 0..(test_arena_chunks() * 2) {
            consumed.push(pool.allocate_for(key.clone()).unwrap());
        }

        // allocate_from_arena(key.clone(), 2) must return a GID from arena 2
        let gid = pool.allocate_from_arena(key.clone(), 2).unwrap();
        assert_eq!(gid.arena_idx(), 2);
        let base2 = (2 * test_gid_stride()) as i64;
        assert!(gid.raw() >= base2);
        assert!(gid.raw() < base2 + test_arena_chunks() as i64);
    }

    #[test]
    fn test_allocate_from_arena_returns_lowest_in_arena() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone()); // idx 0
        pool.register_arena(key.clone()); // idx 1

        // Drain arena 0 so allocate_for would go to arena 1
        let mut g0: Vec<ChunkGid> = (0..test_arena_chunks())
            .map(|_| pool.allocate_for(key.clone()).unwrap())
            .collect();

        // Release g0[3] and g0[7] back â€” both are in arena 0
        let raw3 = g0[3].raw();
        let raw7 = g0[7].raw();
        drop(g0.remove(7));
        drop(g0.remove(3));

        // allocate_from_arena for arena 0 should give back the lowest (raw3)
        let got = pool.allocate_from_arena(key.clone(), 0).unwrap();
        assert_eq!(got.raw(), raw3);
        // second call gives raw7
        let got2 = pool.allocate_from_arena(key.clone(), 0).unwrap();
        assert_eq!(got2.raw(), raw7);
    }

    #[test]
    fn test_allocate_from_arena_empty_returns_none() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone()); // idx 0
        pool.register_arena(key.clone()); // idx 1

        // Drain arena 0 completely
        let _consumed: Vec<ChunkGid> = (0..test_arena_chunks())
            .map(|_| pool.allocate_for(key.clone()).unwrap())
            .collect();

        // Arena 0 is full â€” allocate_from_arena(0) should return None
        assert!(pool.allocate_from_arena(key.clone(), 0).is_none());
    }

    #[test]
    fn test_allocate_from_arena_unregistered_returns_none() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone()); // idx 0 only

        // Arena 99 was never registered
        assert!(pool.allocate_from_arena(key.clone(), 99).is_none());
    }

    #[test]
    fn test_allocate_from_arena_does_not_affect_other_arenas() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone()); // 0
        pool.register_arena(key.clone()); // 1

        let before = pool.free_list_len_for(key.clone());
        let _gid = pool.allocate_from_arena(key.clone(), 1).unwrap();
        assert_eq!(pool.free_list_len_for(key.clone()), before - 1);

        // Arena 0 is completely untouched
        let next = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(next.arena_idx(), 0);
    }

    // â”€â”€ register_arena_at â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_register_arena_at_exact_index() {
        let pool = ChunkGidPool::new();
        let key = float_key();

        // Place at index 5 (skipping 0-4)
        pool.register_arena_at(5, key.clone());

        let gid = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid.arena_idx(), 5);
        let base5 = (5 * test_gid_stride()) as i64;
        assert_eq!(gid.raw(), base5); // lowest slot
    }

    #[test]
    fn test_register_arena_at_fills_free_list() {
        let pool = ChunkGidPool::new();
        let key = float_key();

        pool.register_arena_at(3, key.clone());
        assert_eq!(pool.free_list_len_for(key.clone()), test_arena_chunks());
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn test_register_arena_at_double_panics() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena_at(0, key.clone());
        pool.register_arena_at(0, key.clone()); // should panic
    }

    #[test]
    fn test_register_arena_at_drop_routes_back() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena_at(7, key.clone());

        let gid = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid.arena_idx(), 7);
        drop(gid);

        // Slot returned to the pool â€” free count restored
        assert_eq!(pool.free_list_len_for(key.clone()), test_arena_chunks());
    }

    // â”€â”€ detached GID â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_detached_gid_has_correct_id() {
        let gid = ChunkGid::detached(-1);
        assert_eq!(gid.raw(), -1);
        assert!(gid.is_empty());
    }

    #[test]
    fn test_detached_gid_drop_is_noop_on_live_pool() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone());
        // Allocate from our pool
        let _gid = pool.allocate_for(key.clone()).unwrap();
        let before = pool.free_list_len_for(key.clone());

        // Drop a detached GID that happened to have the same raw id
        let detached = ChunkGid::detached(0);
        drop(detached);

        // The live pool should be completely unaffected
        assert_eq!(pool.free_list_len_for(key.clone()), before);
    }

    // â”€â”€ GID arithmetic helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_gid_arena_idx_and_chunk_idx() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone()); // idx 0
        pool.register_arena(key.clone()); // idx 1

        // Drain arena 0
        let _g0: Vec<ChunkGid> = (0..test_arena_chunks())
            .map(|_| pool.allocate_for(key.clone()).unwrap())
            .collect();

        let gid = pool.allocate_for(key.clone()).unwrap();
        // Should be the first slot of arena 1
        assert_eq!(gid.arena_idx(), 1);
        assert_eq!(gid.chunk_idx(), 0);
        assert_eq!(gid.raw() as usize, test_gid_stride()); // base of arena 1
    }

    #[test]
    fn test_gid_chunk_idx_mid_arena() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone());

        // Skip first 5 slots
        let _skip: Vec<ChunkGid> = (0..5)
            .map(|_| pool.allocate_for(key.clone()).unwrap())
            .collect();
        let gid = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid.arena_idx(), 0);
        assert_eq!(gid.chunk_idx(), 5);
        assert_eq!(gid.raw(), 5);
    }

    // â”€â”€ drop routing correctness per format â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_drop_routes_to_correct_format_pool() {
        let pool = ChunkGidPool::new();
        let float_k = float_key();
        let quant_k = ArenaKey::gpu_quant(crate::kv_cache::QuantFormat::Q8_0);

        pool.register_arena(float_k.clone()); // arena 0 â†’ float
        pool.register_arena(quant_k.clone()); // arena 1 â†’ quant

        let fgid = pool.allocate_for(float_k.clone()).unwrap();
        let qgid = pool.allocate_for(quant_k.clone()).unwrap();
        let f_before = pool.free_list_len_for(float_k.clone());
        let q_before = pool.free_list_len_for(quant_k.clone());

        drop(fgid);
        // Float pool grows, quant pool unchanged
        assert_eq!(pool.free_list_len_for(float_k.clone()), f_before + 1);
        assert_eq!(pool.free_list_len_for(quant_k.clone()), q_before);

        drop(qgid);
        // Quant pool grows, float pool unchanged
        assert_eq!(pool.free_list_len_for(float_k.clone()), f_before + 1);
        assert_eq!(pool.free_list_len_for(quant_k.clone()), q_before + 1);
    }

    // â”€â”€ tombstone threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_tombstone_requires_sufficient_remaining_capacity() {
        let pool = ChunkGidPool::new();
        let key = float_key();

        // Register exactly 2 arenas â€” threshold is 10% of ARENA_CHUNKS (51 free minimum)
        pool.register_arena(key.clone()); // idx 0 â€” fully free (512 slots)
        pool.register_arena(key.clone()); // idx 1 â€” fully free (512 slots)

        // Drain arena 1 down while keeping arena 0 free
        // After taking 512 - 51 = 461 from arena 0 (by draining arena 0 in order)
        // total free = 512 (arena 1) âˆ’ but if we tombstone arena 1 we'd leave only
        // arena 0 which is completely free: 512 â‰¥ 10% of 512 = 51, so tombstone ok.
        // Drain arena 0 slots (lowest GIDs)
        let drained: Vec<ChunkGid> = (0..test_arena_chunks())
            .map(|_| pool.allocate_for(key.clone()).unwrap())
            .collect();
        // now arena 0 fully allocated; arena 1 fully free

        // tombstone arena 1: after release pool would keep arena 0's GIDs on heap
        // but arena 0 is all allocated â€” free after release = 512 - 512 = 0 < 51
        // Wait â€” after tombstone of arena 1, free list shrinks by 512 to 0 < threshold (51).
        let result = pool.next_tombstone(key.clone());
        // Should refuse: after drain remaining free = 0 < ARENA_CHUNKS/10
        assert!(result.is_none());

        // With 3 arenas: after tombstoning one, still 2*512=1024 remaining â‰¥ 51
        drop(drained);
    }

    #[test]
    fn test_tombstone_with_three_arenas_succeeds() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone()); // 0
        pool.register_arena(key.clone()); // 1
        pool.register_arena(key.clone()); // 2

        // Drain arenas 0 all their slots
        let _g0: Vec<ChunkGid> = (0..test_arena_chunks())
            .map(|_| pool.allocate_for(key.clone()).unwrap())
            .collect();

        // Arena 1 and 2 are fully free (1024 total free)
        // Tombstone arena 1: remaining after = 512 â‰¥ 51 âœ“
        let tombstoned = pool.next_tombstone(key.clone()).unwrap();
        assert!(tombstoned == 1 || tombstoned == 2);
        assert_eq!(pool.arena_count(), 2);
        assert_eq!(
            pool.free_list_len_for(key.clone()),
            test_arena_chunks(), // one arena's worth freed
            "tombstoned arena's GIDs should be drained"
        );
    }

    // â”€â”€ exhaustion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_allocate_for_returns_none_when_exhausted() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone());

        // Drain the entire arena
        let mut gids: Vec<ChunkGid> = Vec::new();
        for _ in 0..test_arena_chunks() {
            gids.push(pool.allocate_for(key.clone()).unwrap());
        }

        // Next allocation should return None
        assert!(pool.allocate_for(key.clone()).is_none());

        // After registering another arena â€” works again
        pool.register_arena(key.clone());
        let gid = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid.arena_idx(), 1);
    }

    // â”€â”€ strong_count / is_shared â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_strong_count_tracks_clones() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone());

        let gid = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid.strong_count(), 1);
        assert!(gid.is_unique());
        assert!(!gid.is_shared());

        let clone1 = gid.clone();
        assert_eq!(gid.strong_count(), 2);
        assert!(gid.is_shared());

        let clone2 = gid.clone();
        assert_eq!(gid.strong_count(), 3);

        drop(clone1);
        assert_eq!(gid.strong_count(), 2);
        drop(clone2);
        assert_eq!(gid.strong_count(), 1);
        assert!(gid.is_unique());
    }

    // â”€â”€ equality / identity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_gid_equality_by_id() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone());

        let gid1 = pool.allocate_for(key.clone()).unwrap();
        let gid2 = gid1.clone();
        assert_eq!(gid1, gid2);
        let raw = gid1.raw();

        // Both must be dropped for the id to return to the pool
        drop(gid1);
        drop(gid2); // drops the last Arc; now the id is free again

        let gid3 = pool.allocate_for(key.clone()).unwrap();
        // The pool min-heap returns the lowest freed id â€” that's `raw`
        assert_eq!(gid3.raw(), raw);
    }

    // â”€â”€ register_arena sequential indices â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_register_arena_increments_index() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        for expected in 0..5usize {
            let idx = pool.register_arena(key.clone());
            assert_eq!(idx, expected, "arena index should increment sequentially");
        }
    }

    // â”€â”€ consolidation scenario â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_consolidation_scenario_allocate_from_freed_slot() {
        // Simulate: chunk in arena 2 gets dropped (consolidation moves it to arena 0).
        // Then allocate_from_arena(2) should reclaim the specific slot just freed.
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone()); // 0
        pool.register_arena(key.clone()); // 1
        pool.register_arena(key.clone()); // 2

        // Fill arenas 0 and 1 completely from arena 0 first via normal allocation
        let _g0: Vec<ChunkGid> = (0..test_arena_chunks())
            .map(|_| pool.allocate_for(key.clone()).unwrap())
            .collect();
        let _g1: Vec<ChunkGid> = (0..test_arena_chunks())
            .map(|_| pool.allocate_for(key.clone()).unwrap())
            .collect();

        // Take one slot from arena 2 â€” this is the "source" to be moved
        let src_gid = pool.allocate_from_arena(key.clone(), 2).unwrap();
        let src_raw = src_gid.raw();
        assert_eq!(src_gid.arena_idx(), 2);

        // Simulate consolidation: drop src_gid (data physically moved to arena 0)
        drop(src_gid);

        // The dropped GID returns to arena 2's pool (wrong arena logic here, but
        // let's verify the mechanism: allocate_from_arena on arena 2 reclaims it)
        let reclaimed = pool.allocate_from_arena(key.clone(), 2).unwrap();
        assert_eq!(reclaimed.raw(), src_raw);
        assert_eq!(reclaimed.arena_idx(), 2);
        assert_eq!(reclaimed.chunk_idx(), src_raw as usize % test_gid_stride());
    }

    // â”€â”€ total_free and arena_count â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_total_free_reflects_all_formats() {
        let pool = ChunkGidPool::new();
        let float_k = float_key();
        let quant_k = ArenaKey::gpu_quant(crate::kv_cache::QuantFormat::Q8_0);

        pool.register_arena(float_k.clone());
        pool.register_arena(quant_k.clone());

        let expected_total = test_arena_chunks()
            + arena_chunks_for_format(KvFormat::Quantized(crate::kv_cache::QuantFormat::Q8_0));
        assert_eq!(pool.total_free(), expected_total);

        let _g = pool.allocate_for(float_k.clone()).unwrap();
        assert_eq!(pool.total_free(), expected_total - 1);
    }

    #[test]
    fn test_arena_count_tracks_registrations() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        assert_eq!(pool.arena_count(), 0);
        pool.register_arena(key.clone());
        assert_eq!(pool.arena_count(), 1);
        pool.register_arena(key.clone());
        assert_eq!(pool.arena_count(), 2);
    }

    // â”€â”€ max_gid â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_max_gid_empty_pool() {
        let pool = ChunkGidPool::new();
        assert_eq!(pool.max_gid(), None);
    }

    #[test]
    fn test_max_gid_reflects_highest_arena() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone()); // arena 0: GIDs 0..511
        assert_eq!(pool.max_gid(), Some((test_gid_stride() - 1) as i64));

        pool.register_arena(key.clone()); // arena 1: GIDs 512..1023
        assert_eq!(pool.max_gid(), Some((2 * test_gid_stride() - 1) as i64));
    }
}
