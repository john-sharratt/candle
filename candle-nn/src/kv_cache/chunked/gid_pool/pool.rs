//! The pool of chunk slots for every key, partitioned by format and location,
//! with arena-index recycling.
//!
//! Each key has its own pool, so an allocation never lands in an arena of a
//! different format. An arena index is recycled only after its storage is
//! released, so a new arena cannot adopt its predecessor's stride. An arena made
//! ahead of demand is protected from the empty sweep until its demand arrives or
//! is abandoned, and a creator that never claims closes its window explicitly.

use super::*;

/// Internal state of the GID pool, partitioned by ArenaKey.
pub(super) struct GidPoolState {
    /// arena_idx → ArenaKey registry for routing/compaction bookkeeping.
    /// None = tombstoned (index recycled) or not yet assigned.
    pub(super) arena_registry: Vec<Option<ArenaKey>>,
    /// Monotonic arena index allocator (fallback when `free_arenas` is empty).
    pub(super) next_arena_idx: usize,
    /// FIFO queue of recycled arena indices from tombstoned arenas.
    pub(super) free_arenas: VecDeque<usize>,
    /// Arena indices protected from the empty sweep: arenas created ahead of
    /// a demand that has not filled them yet (`docs/kv_drain_under_pressure.md`
    /// §2). Lifted once the demand arrives or is abandoned.
    pub(super) protected_arenas: AHashSet<usize>,
    /// The classes that keep a **reserve**: one arena's worth of free slots,
    /// which the empty sweep never takes a class below
    /// (`docs/kv_quantize_in_the_gap.md` §3.2). It is the guard against a card
    /// full of native K/V: the gap's first conversion then has somewhere to
    /// write, and every block it installs gives back more native ground than
    /// it took. Everything past it the gap's lease buys between forwards.
    pub(super) reserve_keys: AHashSet<ArenaKey>,
}

/// Inner pool state.
pub(super) struct GidPoolInner {
    /// Immutable key → per-format pool table. Each pool has its own
    /// `RwLock<HashMap>` for register/release; alloc is lock-free.
    pub(super) pools: AHashMap<ArenaKey, ArenaPool>,
    /// Mutable metadata for arena routing/recycling.
    pub(super) metadata: Mutex<GidPoolState>,
    /// Lock-free hint for reclaim detection.
    /// `false` = definitely no fully-free arena.
    /// `true`  = maybe reclaimable; verify under lock.
    pub(super) may_have_reclaimable: AtomicBool,
}

pub(super) fn preallocated_pool_table() -> AHashMap<ArenaKey, ArenaPool> {
    // Locations × size classes — 14 entries, down from ~58 under per-format
    // pools. That collapse is the whole point: every format sharing a class
    // now shares one pool and one free list, so a slot freed by any of them is
    // allocatable by all of them.
    let mut pools = AHashMap::with_capacity(
        ArenaLocation::iter().count() * SizeClass::COUNT + RECORD_STRIDES.len(),
    );
    for location in ArenaLocation::iter() {
        for class in SizeClass::all() {
            pools.insert(ArenaKey::new(class, location), ArenaPool::new(class));
        }
    }
    // **The `KvHead` record pools, preallocated for the same reason — and GPU only.** A
    // record arena's stride is one of the dozen rungs of `RECORD_STRIDES`, which is what
    // keeps this table complete at construction and the lookup lock-free. A record pool
    // sizes its refcount tables from the key's own capacity, not from a size class,
    // because a record's slot count comes from its stride
    // (`docs/vram_span_partition.md` §8).
    //
    // Not per location: records are allocated `Gpu` only, and a record cannot migrate
    // tiers because its device address is fixed for the handle's life — every slice
    // header holds it raw. A per-location loop minted twelve CPU pools that nothing
    // could ever allocate from.
    for stride in RECORD_STRIDES {
        let key = ArenaKey::for_record_stride(ArenaLocation::Gpu, stride);
        pools.insert(key, ArenaPool::with_chunks(key.chunks()));
    }
    pools
}

/// Pool for allocating and recycling gids, partitioned by ArenaKey.
pub struct ChunkGidPool {
    pub(super) inner: Arc<GidPoolInner>,
}

impl ChunkGidPool {
    /// Create a new GID pool with no registered arenas.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(GidPoolInner {
                pools: preallocated_pool_table(),
                metadata: Mutex::new(GidPoolState {
                    arena_registry: Vec::with_capacity(64),
                    next_arena_idx: 0,
                    free_arenas: VecDeque::with_capacity(16),
                    protected_arenas: AHashSet::with_capacity(32),
                    reserve_keys: AHashSet::with_capacity(16),
                }),
                may_have_reclaimable: AtomicBool::new(false),
            }),
        }
    }

    /// Mark an arena index as protected from compaction tombstoning.
    pub fn protect_arena(&self, arena_idx: usize) {
        let mut state = self.inner.metadata.lock().unwrap();
        state.protected_arenas.insert(arena_idx);
    }

    /// Keep a reserve for `key`: one arena's worth of free slots.
    pub fn keep_reserve(&self, key: ArenaKey) {
        let mut state = self.inner.metadata.lock().unwrap();
        state.reserve_keys.insert(key);
    }

    /// Keep no reserve for any class — for a backing whose classes were named
    /// at a width it no longer writes.
    pub fn clear_reserves(&self) {
        let mut state = self.inner.metadata.lock().unwrap();
        state.reserve_keys.clear();
    }

    /// The classes that keep a reserve.
    pub fn reserve_keys(&self) -> Vec<ArenaKey> {
        let state = self.inner.metadata.lock().unwrap();
        state.reserve_keys.iter().copied().collect()
    }

    /// The free slots `key`'s reserve holds — one arena's worth — or `None`
    /// for a class that keeps no reserve.
    pub fn reserve_target(&self, key: ArenaKey) -> Option<usize> {
        let state = self.inner.metadata.lock().unwrap();
        state.reserve_keys.contains(&key).then(|| key.chunks())
    }

    /// Slots live across `key`'s arenas — one atomic read.
    pub fn live_slots(&self, key: ArenaKey) -> usize {
        self.inner
            .pools
            .get(&key)
            .map_or(0, |pool| pool.total_live())
    }

    /// Slots free across `key`'s arenas.
    pub fn free_slots(&self, key: ArenaKey) -> usize {
        self.pool_arena_load(key)
            .iter()
            .map(|&(_, capacity, live)| capacity.saturating_sub(live))
            .sum()
    }

    /// Lift [`Self::protect_arena`], so the empty sweep may reclaim the arena again.
    pub fn unprotect_arena(&self, arena_idx: usize) {
        let mut state = self.inner.metadata.lock().unwrap();
        state.protected_arenas.remove(&arena_idx);
    }

    /// Register a new arena with the pool.
    ///
    /// Recycles a tombstoned arena index (FIFO) or assigns the next
    /// fresh index. Creates the per-arena refcount table and returns
    /// the assigned arena index.
    pub fn register_arena(&self, key: ArenaKey) -> usize {
        let arena_idx = {
            let mut state = self.inner.metadata.lock().unwrap();
            let arena_idx = state.free_arenas.pop_front().unwrap_or_else(|| {
                let idx = state.next_arena_idx;
                state.next_arena_idx += 1;
                idx
            });
            if arena_idx >= state.arena_registry.len() {
                state.arena_registry.resize(arena_idx + 1, None);
            }
            state.arena_registry[arena_idx] = Some(key);
            arena_idx
        };
        let pool = self
            .inner
            .pools
            .get(&key)
            .expect("register_arena: missing preallocated pool for key");
        pool.register_arena(arena_idx, key);
        self.inner
            .may_have_reclaimable
            .store(true, Ordering::Release);
        arena_idx
    }

    /// Register an externally-created arena at a specific arena index.
    ///
    /// This is used when chunk_ops creates a quant or float arena at a
    /// known index (e.g., after `ArenaStorageState::arenas.len()`
    /// grows). Unlike `register_arena` which assigns the next free
    /// index, this registers at exactly the given index — necessary
    /// when ArenaStorageState and GidPool must agree on the arena
    /// index.
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
            state.arena_registry[arena_idx] = Some(key);
        }
        let pool = self
            .inner
            .pools
            .get(&key)
            .expect("register_arena_at: missing preallocated pool for key");
        pool.register_arena(arena_idx, key);
        self.inner
            .may_have_reclaimable
            .store(true, Ordering::Release);
    }

    /// Record that `key`'s arena `arena_idx` is carved from region `rank` — its
    /// place in the reservation, which every allocation and every release of the
    /// arena is ordered by. Called once the region is claimed.
    pub fn set_arena_rank(&self, key: ArenaKey, arena_idx: usize, rank: usize) {
        if let Some(pool) = self.inner.pools.get(&key) {
            pool.set_rank(arena_idx, rank);
        }
    }

    /// Allocate a gid for the given format.
    ///
    /// Takes the lowest-placed arena with a free slot. Returns
    /// `None` if no arena of this format has free capacity — caller
    /// should `register_arena` and retry.
    pub fn allocate_for(&self, key: ArenaKey) -> Option<ChunkGid> {
        let pool = self.inner.pools.get(&key)?;
        let (id, table) = pool.allocate_any()?;
        Some(ChunkGid {
            id,
            backing: GidBacking::Pooled(table),
        })
    }

    /// Allocate `len` CONSECUTIVE slots in one arena of `key`'s pool. `None`
    /// when no arena's never-used tail fits the run — the caller registers a
    /// fresh arena and retries (a fresh arena always fits: `len` ≤ capacity).
    pub fn allocate_run_for(&self, key: ArenaKey, len: usize) -> Option<Vec<ChunkGid>> {
        let pool = self.inner.pools.get(&key)?;
        let (first, table) = pool.allocate_run(len)?;
        Some(
            (0..len as i64)
                .map(|i| ChunkGid {
                    id: first + i,
                    backing: GidBacking::Pooled(Arc::clone(&table)),
                })
                .collect(),
        )
    }

    /// The arena a run claim of `len` slots in `key`'s pool starts its walk at.
    #[cfg(test)]
    pub(super) fn run_floor_for(&self, key: ArenaKey, len: usize) -> usize {
        self.inner
            .pools
            .get(&key)
            .map_or(0, |pool| pool.run_floor(len).load(Ordering::Relaxed))
    }

    /// Whether a run of `len` consecutive slots could be claimed right now,
    /// **without claiming it**.
    ///
    /// A run claim advances an arena's never-used high-water mark
    /// irreversibly — dropped run gids recycle through the singleton free
    /// stack, which `try_claim_run` never reads — so an allocate-and-drop
    /// "probe" permanently burns `len` slots of contiguous capacity. This is
    /// the read-only question that probe was trying to ask.
    pub fn run_would_fit(&self, key: ArenaKey, len: usize) -> bool {
        let Some(pool) = self.inner.pools.get(&key) else {
            return false;
        };
        let tables = pool.tables.read().unwrap();
        let order = pool.order.read().unwrap();
        let floor = pool.run_floor(len).load(Ordering::Relaxed);
        order
            .range(floor..)
            .any(|(_, idx)| tables.get(idx).is_some_and(|t| t.run_fits(len)))
    }

    /// [`Self::allocate_run_for`] against one specific arena index — see
    /// `ChunkPool::allocate_run_in` for why the caller targets the arena it
    /// just registered instead of re-walking.
    pub fn allocate_run_for_in(
        &self,
        key: ArenaKey,
        arena_idx: usize,
        len: usize,
    ) -> Option<Vec<ChunkGid>> {
        let pool = self.inner.pools.get(&key)?;
        let (first, table) = pool.allocate_run_in(arena_idx, len)?;
        Some(
            (0..len as i64)
                .map(|i| ChunkGid {
                    id: first + i,
                    backing: GidBacking::Pooled(Arc::clone(&table)),
                })
                .collect(),
        )
    }

    /// Bulk variant of [`Self::allocate_for`] — returns up to `n` gids.
    ///
    /// May return fewer than `n` if the pool ran out of capacity; the
    /// caller registers a fresh arena and re-invokes to fill the
    /// remainder, exactly mirroring the singular code path's
    /// `register_arena + retry`.
    pub fn allocate_n_for(&self, key: ArenaKey, n: usize) -> Vec<ChunkGid> {
        let Some(pool) = self.inner.pools.get(&key) else {
            return Vec::new();
        };
        pool.allocate_n(n)
            .into_iter()
            .map(|(id, table)| ChunkGid {
                id,
                backing: GidBacking::Pooled(table),
            })
            .collect()
    }

    /// Allocate a gid from a specific arena (for consolidation that
    /// must target a particular destination arena). Returns `None` if
    /// that arena is full or not registered.
    pub fn allocate_from_arena(&self, key: ArenaKey, target_arena: usize) -> Option<ChunkGid> {
        let pool = self.inner.pools.get(&key)?;
        let (id, table) = pool.allocate_from_arena(target_arena)?;
        Some(ChunkGid {
            id,
            backing: GidBacking::Pooled(table),
        })
    }

    /// Per-arena occupancy for one pool — the gid-pool half of a compaction
    /// census.
    ///
    /// Returns `(arena_idx, capacity, occupied slot indices ascending)` for every
    /// registered arena of `key`, itself in ascending `arena_idx` order because
    /// that is the `BTreeMap`'s order.
    ///
    /// **`arena_idx` order is not address order.** An index is whatever the pool
    /// handed out and is recycled when an arena is tombstoned, so the caller must
    /// join this against each arena's region index before planning anything —
    /// packing by index instead of address is the mistake
    /// `compact_plan::ArenaSlots::rank` exists to prevent, and it silently packs
    /// into the wrong end of the span.
    /// `(arena_idx, capacity, live_count)` for every registered arena of `key` —
    /// the **cheap** form of [`Self::pool_occupancy`].
    ///
    /// Occupancy's cost is `occupied_slots()`, which walks an arena's whole refcount
    /// bitmap and materialises a `Vec<u32>` of its live slots: at the 320 B rung that
    /// is 52,428 bits and up to 52,428 `u32`s per arena. A caller that only needs to
    /// know *how much* is live — to decide which pool is worth censusing at all —
    /// must not pay that, or the decision costs more than the thing it is deciding
    /// about.
    pub fn pool_arena_load(&self, key: ArenaKey) -> Vec<(usize, usize, usize)> {
        let Some(pool) = self.inner.pools.get(&key) else {
            return Vec::new();
        };
        let tables = pool.tables.read().unwrap();
        tables
            .iter()
            .map(|(&arena_idx, t)| (arena_idx, pool.arena_chunks, t.live_count()))
            .collect()
    }

    /// Arenas one pool holds, and how many it would need packed — the **cheap**
    /// sparsity figure.
    ///
    /// `Fragmentation::packed_arenas` computes the same number from a full occupancy
    /// census, and the difference matters because this one is a gate: a pass that
    /// censuses the ladder to decide whether to census the ladder has already paid.
    /// Every term here is a field read — the live count is an atomic the refcount
    /// table maintains — so an idle engine can ask this on every wave.
    ///
    /// `(arenas_held, arenas_if_packed)`.
    pub fn pool_sparsity(&self, key: ArenaKey) -> (usize, usize) {
        let load = self.pool_arena_load(key);
        if load.is_empty() {
            return (0, 0);
        }
        let capacity = load[0].1.max(1);
        let live: usize = load.iter().map(|(_, _, live)| *live).sum();
        (load.len(), live.div_ceil(capacity))
    }

    pub fn pool_occupancy(&self, key: ArenaKey) -> Vec<(usize, usize, Vec<u32>)> {
        let Some(pool) = self.inner.pools.get(&key) else {
            return Vec::new();
        };
        let tables = pool.tables.read().unwrap();
        tables
            .iter()
            .map(|(&arena_idx, t)| (arena_idx, pool.arena_chunks, t.occupied_slots()))
            .collect()
    }

    /// Convenience: allocate a gid using a default test key.
    pub fn allocate(&self) -> ChunkGid {
        let key = ArenaKey::new(TEST_CLASS, ArenaLocation::Gpu);
        if let Some(gid) = self.allocate_for(key) {
            return gid;
        }
        self.register_arena(key);
        self.allocate_for(key)
            .expect("just registered arena, must have capacity")
    }

    /// Find a fully-free arena of this format and release its pool entry.
    ///
    /// **The index is not reissued yet.** The caller releases the arena's storage
    /// next and then hands the index back with [`Self::recycle_arena_index`]. Freed
    /// here, a registration on another thread could take the index while storage
    /// still held the old arena, and `ensure_arena_exists` — which asks only whether
    /// an arena exists at the index — would adopt it under the new key with the old
    /// key's stride.
    pub fn next_tombstone(&self, key: ArenaKey, lowest_hole: Option<usize>) -> Option<usize> {
        let mut state = self.inner.metadata.lock().unwrap();
        let pool = self.inner.pools.get(&key)?;
        let reserve = state.reserve_keys.contains(&key);
        let arena_idx = pool.try_tombstone(&state.protected_arenas, reserve, lowest_hole)?;
        state.arena_registry[arena_idx] = None;
        Some(arena_idx)
    }

    /// Return a tombstoned index to the free list, once its storage is released.
    ///
    /// The second half of [`Self::next_tombstone`] and [`Self::tombstone_if_empty`]:
    /// from here the next registration may take the index, and storage holds no
    /// arena at it to be mistaken for the new one.
    pub fn recycle_arena_index(&self, arena_idx: usize) {
        let mut state = self.inner.metadata.lock().unwrap();
        debug_assert!(
            state
                .arena_registry
                .get(arena_idx)
                .is_none_or(|k| k.is_none()),
            "recycling arena index {arena_idx} that is still registered"
        );
        state.free_arenas.push_back(arena_idx);
    }

    /// Release one named arena's pool entry if it is empty and past its creation
    /// window; `true` when it did.
    ///
    /// For a creator handing back an arena it made and did not use. The caller releases
    /// the arena's storage afterwards and then recycles the index, exactly as after
    /// [`Self::next_tombstone`].
    ///
    /// **The index leaves the protected set here, under the same lock that puts it on
    /// the free list.** Lifted any later, the index would sit free and protected at once,
    /// and a creator that registered and protected it in between would have its
    /// protection erased by the belated lift.
    pub fn tombstone_if_empty(&self, key: ArenaKey, arena_idx: usize) -> bool {
        let mut state = self.inner.metadata.lock().unwrap();
        let Some(pool) = self.inner.pools.get(&key) else {
            return false;
        };
        if !pool.tombstone_if_empty(arena_idx) {
            return false;
        }
        state.arena_registry[arena_idx] = None;
        state.protected_arenas.remove(&arena_idx);
        true
    }

    /// Declare an arena created **ahead of demand** finished, so the empty
    /// sweep may reclaim it once the demand it was stamped for goes away.
    ///
    /// Every other creator closes this window by allocating its first chunk.
    /// See [`ArenaRefcounts::end_creation_window`] for why the pre-creation
    /// path cannot, and what it costs when the window is left open.
    pub fn finish_creation(&self, key: ArenaKey, arena_idx: usize) {
        if let Some(pool) = self.inner.pools.get(&key) {
            pool.finish_creation(arena_idx);
        }
    }

    /// Remove from `free_arenas` any indices >= `threshold`.
    pub fn drain_free_arenas_above(&self, threshold: usize) {
        let mut state = self.inner.metadata.lock().unwrap();
        state.free_arenas.retain(|&idx| idx < threshold);
    }

    /// Check whether any arena is fully free.
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

    /// Every arena holding no chunk, across every key, and why the empty sweep would
    /// keep any of them — what an out-of-regions report needs before it can claim the
    /// reservation is occupied.
    pub fn empty_census(&self) -> EmptyCensus {
        let state = self.inner.metadata.lock().unwrap();
        let mut census = EmptyCensus::default();
        for pool in self.inner.pools.values() {
            pool.count_empty(&state.protected_arenas, &mut census);
        }
        census
    }

    /// Every key that has an arena, with its arena count and live slots, most arenas
    /// first — what is holding the reservation at the moment it is read.
    pub fn occupancy(&self) -> Vec<KeyOccupancy> {
        let mut out: Vec<KeyOccupancy> = self
            .inner
            .pools
            .iter()
            .map(|(key, pool)| KeyOccupancy {
                key: *key,
                arenas: pool.total_arenas.load(Ordering::Relaxed),
                live: pool.total_live(),
                arena_chunks: pool.arena_chunks,
            })
            .filter(|o| o.arenas > 0)
            .collect();
        out.sort_by_key(|o| Reverse(o.arenas));
        out
    }

    /// True when a forced compaction could free at least one whole arena across
    /// any registered pool. When false, every pool is packed to within a single
    /// arena of free space — compaction would reclaim nothing, so callers under
    /// VRAM pressure should skip the (expensive) pass rather than spin on it.
    pub fn can_reclaim_arena(&self) -> bool {
        self.inner
            .pools
            .values()
            .any(|pool| pool.reclaimable_arenas() >= 1)
    }

    /// Force-release the bookkeeping of an arena whose storage was never created —
    /// the creation paths' rollback when building the arena failed — and recycle
    /// its index at once, since storage holds nothing at it.
    ///
    /// **The pool entry goes before the index is freed, under one lock.** Freed
    /// first, a registration on another thread could take the index and land its
    /// key's entry beside the stale one, and this removal would then take the new
    /// tenant's.
    pub fn force_release_arena(&self, arena_idx: usize) {
        let mut state = self.inner.metadata.lock().unwrap();
        let Some(key) = state.arena_registry.get(arena_idx).and_then(|k| *k) else {
            return;
        };
        if let Some(pool) = self.inner.pools.get(&key) {
            pool.force_release(arena_idx);
        }
        state.arena_registry[arena_idx] = None;
        state.free_arenas.push_back(arena_idx);
    }

    /// Return the set of format keys currently registered in the pool.
    pub fn format_keys(&self) -> Vec<ArenaKey> {
        let state = self.inner.metadata.lock().unwrap();
        let mut out: AHashSet<ArenaKey> = AHashSet::new();
        for key in state.arena_registry.iter().flatten() {
            out.insert(*key);
        }
        out.into_iter().collect()
    }

    /// Maximum gid currently in circulation (for compaction bound checks).
    pub fn max_gid(&self) -> Option<i64> {
        let state = self.inner.metadata.lock().ok()?;
        let mut max: i64 = -1;
        for (idx, entry) in state.arena_registry.iter().enumerate() {
            if entry.is_some() {
                let arena_top = ((idx + 1) * GID_STRIDE) as i64 - 1;
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

    /// Resync diagnostic counters from authoritative table state.
    /// With the new lock-free design the counters are inherently
    /// up-to-date — this method is a no-op kept for API compatibility.
    pub(crate) fn resync_counters(&self) {}

    /// Number of free gids for a specific arena index. Returns 0 if
    /// the arena isn't registered or has no free gids.
    pub(crate) fn arena_free_count(&self, arena_idx: usize) -> u32 {
        let state = self.inner.metadata.lock().unwrap();
        let key = match state.arena_registry.get(arena_idx) {
            Some(Some(k)) => *k,
            _ => return 0,
        };
        drop(state);
        self.inner
            .pools
            .get(&key)
            .map(|p| p.free_count_for_arena(arena_idx))
            .unwrap_or(0)
    }

    /// Free-list length for a specific format — used by the
    /// pool/lifecycle tests in this module. Computed as
    /// `total_arenas * arena_chunks - total_live`.
    #[cfg(test)]
    pub(crate) fn free_list_len_for(&self, key: ArenaKey) -> usize {
        self.inner
            .pools
            .get(&key)
            .map(|p| p.free_len())
            .unwrap_or(0)
    }

    /// Sum of live gids across all formats — for diagnostics.
    #[allow(dead_code)]
    pub(crate) fn total_live(&self) -> usize {
        self.inner
            .pools
            .values()
            .map(|pool| pool.total_live())
            .sum()
    }

    /// GPU arena occupancy per size class. Reads the lock-free per-pool
    /// atomics (`O(classes × locations)` = 14 entries, most empty) — cheap
    /// enough for the per-wave `kv-pool` diagnostic. See
    /// [`GpuArenaClassStats`].
    pub(crate) fn gpu_class_stats(&self) -> GpuArenaClassStats {
        let mut s = GpuArenaClassStats::default();
        for (i, c) in s.classes.iter_mut().enumerate() {
            c.slot_bytes = SizeClass::from_index(i).map_or(0, |cl| cl.bytes());
        }
        for (key, pool) in self.inner.pools.iter() {
            if key.location != ArenaLocation::Gpu {
                continue;
            }
            // **Record arenas are not a size class and must not be counted as one.**
            // This walk is over `pools.iter()`, not over band keys, so unlike the census
            // in `compact` it does see the record pools — and a record key carries
            // `SizeClass::at(0)` only to keep the key one shape, so all twelve record
            // rungs would land in the smallest class's row. That row then reports arenas
            // it does not own, reserved bytes for regions that are not its, and
            // `live_bytes` accumulated at strides up to 1 MiB against a 320 B
            // `slot_bytes` — and it feeds the fragmentation and relief decisions.
            if !matches!(key.kind, ArenaKind::Band) {
                continue;
            }
            let arenas = pool.total_arenas.load(Ordering::Relaxed);
            if arenas == 0 {
                continue;
            }
            let row = &mut s.classes[key.class.index()];
            row.arenas += arenas;
            row.reserved_bytes += arenas.saturating_mul(TARGET_ARENA_BYTES);
            // Every slot in a class costs its stride, whatever occupies it.
            row.live_bytes += pool.total_live().saturating_mul(key.slot_stride());
        }
        s
    }
}

impl Default for ChunkGidPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ChunkGidPool {
    /// Clone the pool — shares the same internal state.
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
