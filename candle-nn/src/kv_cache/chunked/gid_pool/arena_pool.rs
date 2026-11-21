//! The per-format pool of arenas, each with a lock-free refcount table, in
//! placement order.
//!
//! A claim walks the arenas lowest-placed first, so the span fills from its low
//! end and compaction can pack against it. The capacity bitmap finds that arena
//! directly, and a stale bit is cleared when it is seen rather than trusted. Run
//! claims start from per-length floors that skip the prefix of arenas too short
//! to hold them.

use super::*;

/// Per-format arena pool: refcount tables keyed by arena_idx.
///
/// Each arena owns a lock-free [`ArenaRefcounts`] table behind an `Arc`.
/// The pool only takes its `RwLock` on `register_arena` / `release_arena`
/// Places a ranked arena can take — one per region of the largest span the pool
/// is sized for: a 512 GiB VRAM ceiling over the arena size, far past any single
/// card. Region ranks and `arena_idx` (VRAM-bounded, recycled on tombstone) both
/// stay below it.
pub(super) const RANKED_PLACES: usize = {
    let regions = (512usize << 30) / TARGET_ARENA_BYTES;
    if regions > 4096 {
        regions
    } else {
        4096
    }
};

/// The place of an arena with no region rank: after every ranked place, in
/// index order.
#[inline]
pub(super) const fn unranked_place(arena_idx: usize) -> usize {
    RANKED_PLACES + arena_idx
}

/// Per-format bitmap: bit `p` set ⇒ the arena at place `p` has ≥1 free slot.
/// The pool's O(1) "which arena has capacity" index — it replaces the per-alloc
/// walk over every arena that was the pressure-regime `alloc` bottleneck
/// (`allocate_any` used to iterate the whole `tables` map, O(num_arenas), for
/// every one of the ~1M allocations a drain pass issues). Fully lock-free: `dec`
/// sets a bit on a full→non-full transition, alloc clears it on non-full→full
/// and reads it via find-first-set. Fixed-size so `dec` never contends a resize:
/// [`RANKED_PLACES`] ranked places, then as many unranked ones. `find-first-set`
/// returns the lowest place — the arena lowest in the span — which is the
/// lowest-first packing compaction relies on.
#[derive(Debug)]
pub(super) struct CapacityBitmap {
    pub(super) words: Box<[AtomicU64]>,
}

impl CapacityBitmap {
    pub(super) fn new() -> Self {
        let words = (0..(2 * RANKED_PLACES).div_ceil(64))
            .map(|_| AtomicU64::new(0))
            .collect();
        Self { words }
    }

    #[inline]
    pub(super) fn set(&self, place: usize) {
        if let Some(w) = self.words.get(place >> 6) {
            w.fetch_or(1u64 << (place & 63), Ordering::Release);
        }
    }

    #[inline]
    pub(super) fn clear(&self, place: usize) {
        if let Some(w) = self.words.get(place >> 6) {
            w.fetch_and(!(1u64 << (place & 63)), Ordering::Release);
        }
    }

    /// Lowest place whose has-capacity bit is set, or `None` if all clear.
    #[inline]
    pub(super) fn first_set(&self) -> Option<usize> {
        for (wi, w) in self.words.iter().enumerate() {
            let bits = w.load(Ordering::Acquire);
            if bits != 0 {
                return Some(wi * 64 + bits.trailing_zeros() as usize);
            }
        }
        None
    }
}

/// (rare); allocation walks the tables read-locked and operates lock-
/// free against the chosen arena's counts.
#[derive(Debug)]
pub(super) struct ArenaPool {
    /// `arena_idx → refcount table`. `RwLock`'d for register/release;
    /// reads are uncontended in steady state.
    pub(super) tables: RwLock<BTreeMap<usize, Arc<ArenaRefcounts>>>,
    /// `place → arena_idx` for every arena in `tables`: the placement order the
    /// allocators walk and the empty sweep chooses by, lowest in the span
    /// first, kept so neither rebuilds and sorts it per claim. Taken after
    /// `tables`, never before.
    pub(super) order: RwLock<BTreeMap<usize, usize>>,
    /// Number of currently-registered arenas for this key. Lock-free
    /// counter for fast diagnostics.
    pub(super) total_arenas: AtomicUsize,
    /// Running count of currently-allocated slots across every arena
    /// in this per-format pool. Updated by [`ArenaRefcounts`] on each
    /// `0 → 1` / `1 → 0` transition, so `total_live()` is an O(1)
    /// atomic load. Shared (`Arc`) with every refcount table.
    pub(super) total_live: Arc<AtomicUsize>,
    /// Physical chunk capacity for one arena of this specific format.
    pub(super) arena_chunks: usize,
    /// Serializes the *claiming* walk (`allocate_any` / `allocate_n` /
    /// `allocate_excluding`). Only one thread scans this pool's `counts`
    /// arrays for a free slot at a time, so the scan reads a slot's
    /// occupancy with a plain relaxed load (no per-slot locked RMW) and the
    /// 128 worker/persist threads stop ping-ponging the same cache lines.
    /// Drops (`ArenaRefcounts::dec`) are deliberately NOT gated — they run
    /// lock-free on arbitrary threads; the gate only removes claimer↔claimer
    /// contention. Held for bookkeeping only, never across GPU work.
    pub(super) alloc_gate: Mutex<()>,
    /// Bit `p` set ⇒ the arena at place `p` has ≥1 free slot. `allocate_any`
    /// finds the lowest such arena via find-first-set instead of walking every
    /// arena.
    pub(super) capacity: Arc<CapacityBitmap>,
    /// `run_floors[l]`: every arena placed below this place has a never-used tail
    /// shorter than `l` slots, so it can host no run of `l` or more — its
    /// high-water mark never retreats. A run claim of length `l` starts at
    /// `run_floors[min(l, RUN_FLOOR_LENS − 1)]` instead of at the lowest arena,
    /// which keeps lowest-first placement while skipping the prefix that cannot
    /// serve it: that prefix is every arena a wide prefill has already filled,
    /// and a chunk makes 2·n_kv_head run claims, so walking it each time made a
    /// claim's cost grow with the arenas in the pool. One floor per length,
    /// because an arena whose tail is a few slots short of a run — left behind
    /// when single-slot claims have moved its high-water mark off the run grid —
    /// stops a floor kept for "no tail at all" dead, and every run claim then
    /// walked every arena past it. Lowered when an arena takes a place below them
    /// (registered, or ranked into a lower place, with a whole tail); only moved
    /// under the `tables` lock.
    pub(super) run_floors: [AtomicUsize; RUN_FLOOR_LENS],
}

/// Run lengths with a floor of their own in [`ArenaPool::run_floors`]; a longer
/// run starts from the last one's, which no longer run can be served below.
pub(super) const RUN_FLOOR_LENS: usize = 33;

impl ArenaPool {
    pub(super) fn new(class: SizeClass) -> Self {
        Self::with_chunks(class.chunks_per_region())
    }

    /// A pool whose arenas hold `arena_chunks` slots.
    ///
    /// Split out because a record arena's capacity comes from its stride, not from its
    /// size class — `ArenaKey::chunks()` is the authority for both kinds, and taking it
    /// from the class for a record pool would size every refcount table wrongly.
    pub(super) fn with_chunks(arena_chunks: usize) -> Self {
        Self {
            tables: RwLock::new(BTreeMap::new()),
            order: RwLock::new(BTreeMap::new()),
            total_arenas: AtomicUsize::new(0),
            total_live: Arc::new(AtomicUsize::new(0)),
            arena_chunks,
            alloc_gate: Mutex::new(()),
            capacity: Arc::new(CapacityBitmap::new()),
            run_floors: std::array::from_fn(|_| AtomicUsize::new(0)),
        }
    }

    /// The floor a run of `len` slots starts its walk at.
    pub(super) fn run_floor(&self, len: usize) -> &AtomicUsize {
        &self.run_floors[len.min(RUN_FLOOR_LENS - 1)]
    }

    /// Register a new arena with the pool — creates its refcount table.
    pub(super) fn register_arena(&self, arena_idx: usize, key: ArenaKey) -> Arc<ArenaRefcounts> {
        let table = Arc::new(ArenaRefcounts::new(
            self.arena_chunks,
            arena_idx,
            key,
            Arc::clone(&self.total_live),
            Arc::clone(&self.capacity),
        ));
        let place = table.place();
        {
            let mut tables = self.tables.write().unwrap();
            tables.insert(arena_idx, Arc::clone(&table));
            self.order.write().unwrap().insert(place, arena_idx);
            // A fresh arena has its whole tail: runs of every length may land
            // at or above it.
            for floor in &self.run_floors {
                floor.fetch_min(place, Ordering::Relaxed);
            }
        }
        self.total_arenas.fetch_add(1, Ordering::Relaxed);
        // A fresh arena is all free — mark it available. Ordered after the
        // `tables` insert so a claimer that sees the bit also finds the table.
        self.capacity.set(place);
        table
    }

    /// Record that `arena_idx`'s slab is carved from region `rank`, moving it to
    /// that place in the placement order.
    ///
    /// **After the region is claimed, never at registration**: the index is handed
    /// out before its storage exists, and recycled onto a different region once the
    /// arena is tombstoned. The rank goes with the refcount table, which the
    /// tombstone drops, so a recycled index starts unranked again.
    ///
    /// Under the claim gate, so no claimer reads the arena's capacity bit while it
    /// moves; a drop racing the move can leave a stale bit at the old place, which
    /// [`Self::allocate_any`] clears on sight.
    pub(super) fn set_rank(&self, arena_idx: usize, rank: usize) {
        if rank >= RANKED_PLACES {
            return;
        }
        let _gate = self.alloc_gate.lock().unwrap();
        let tables = self.tables.read().unwrap();
        let Some(table) = tables.get(&arena_idx) else {
            return;
        };
        let old = table.place.swap(rank, Ordering::AcqRel);
        if old == rank {
            return;
        }
        {
            let mut order = self.order.write().unwrap();
            order.remove(&old);
            order.insert(rank, arena_idx);
        }
        self.capacity.clear(old);
        if !table.is_full() {
            self.capacity.set(rank);
        }
        for floor in &self.run_floors {
            floor.fetch_min(rank, Ordering::Relaxed);
        }
    }

    /// Drop `arena_idx`'s table and its place, and clear its capacity bit;
    /// `true` when there was one. The caller holds the `tables` write lock.
    pub(super) fn remove_table(
        &self,
        tables: &mut BTreeMap<usize, Arc<ArenaRefcounts>>,
        arena_idx: usize,
    ) -> bool {
        let Some(table) = tables.remove(&arena_idx) else {
            return false;
        };
        let place = table.place();
        self.order.write().unwrap().remove(&place);
        self.total_arenas.fetch_sub(1, Ordering::Relaxed);
        self.capacity.clear(place);
        true
    }

    /// Claim `len` CONSECUTIVE slots in one arena of this pool, returning the
    /// first raw gid and the arena's refcount table. Only the never-used
    /// high-water tail of an arena can host a run (see
    /// [`ArenaRefcounts::try_claim_run`]); arenas whose tail is exhausted are
    /// skipped, and `None` means the caller must register a fresh arena.
    pub(super) fn allocate_run(&self, len: usize) -> Option<(i64, Arc<ArenaRefcounts>)> {
        let _gate = self.alloc_gate.lock().unwrap();
        let stride = GID_STRIDE;
        let tables = self.tables.read().unwrap();
        let order = self.order.read().unwrap();
        let floor_len = len.min(RUN_FLOOR_LENS - 1);
        let floor = self.run_floor(len).load(Ordering::Relaxed);
        // The prefix that cannot serve this length grows as the walk passes
        // arenas whose tail is too short for it, and stops at the first arena
        // that could. Only `floor_len`'s floor moves: a tail too short for a
        // longer run may still serve a shorter one, and one too short for a run
        // past the last floor says nothing about that floor's length.
        let mut next_floor = floor;
        let mut prefix_exhausted = true;
        for (&place, arena_idx) in order.range(floor..) {
            let Some(table) = tables.get(arena_idx) else {
                continue;
            };
            if table.run_fits(len) {
                if let Some(first) = table.try_claim_run(len) {
                    if table.is_full() {
                        self.capacity.clear(place);
                    }
                    self.run_floors[floor_len].store(next_floor, Ordering::Relaxed);
                    return Some(((arena_idx * stride + first) as i64, Arc::clone(table)));
                }
            }
            if prefix_exhausted && !table.run_fits(floor_len) {
                next_floor = place + 1;
            } else {
                prefix_exhausted = false;
            }
        }
        self.run_floors[floor_len].store(next_floor, Ordering::Relaxed);
        None
    }

    /// Claim a run from ONE SPECIFIC arena — the freshly registered one.
    ///
    /// The global [`Self::allocate_run`] walk cannot promise anything about a
    /// fresh arena: between the caller's `register_arena` and its retry walk,
    /// any other gated claimer (24-way parallel elevation of the same format is
    /// routine) can consume the new arena's high-water tail, and the retry then
    /// fails as if no space existed. Claiming by index removes the "which arena"
    /// race — the only way THIS can fail is racers landing in the same arena,
    /// which the caller answers by registering another (bounded loop).
    pub(super) fn allocate_run_in(
        &self,
        arena_idx: usize,
        len: usize,
    ) -> Option<(i64, Arc<ArenaRefcounts>)> {
        // Same gate as every claim walk: `try_claim_run` is load-then-store on
        // `hwm` and is only sound serialized.
        let _gate = self.alloc_gate.lock().unwrap();
        let stride = GID_STRIDE;
        let tables = self.tables.read().unwrap();
        let table = tables.get(&arena_idx)?;
        let first = table.try_claim_run(len)?;
        if table.is_full() {
            self.capacity.clear(table.place());
        }
        Some(((arena_idx * stride + first) as i64, Arc::clone(table)))
    }

    /// The arena at `place`, if one stands there: its index and table.
    pub(super) fn at_place<'t>(
        tables: &'t BTreeMap<usize, Arc<ArenaRefcounts>>,
        order: &BTreeMap<usize, usize>,
        place: usize,
    ) -> Option<(usize, &'t Arc<ArenaRefcounts>)> {
        let arena_idx = *order.get(&place)?;
        tables
            .get(&arena_idx)
            .filter(|t| t.place() == place)
            .map(|t| (arena_idx, t))
    }

    pub(super) fn allocate_any(&self) -> Option<(i64, Arc<ArenaRefcounts>)> {
        // Serialize the claiming walk: only one thread scans this pool's
        // `counts` arrays at a time, so `try_claim_one` can probe occupancy
        // with a relaxed load (no per-slot locked RMW) without 128 threads
        // ping-ponging the same cache lines. Drops stay lock-free.
        let _gate = self.alloc_gate.lock().unwrap();
        let stride = GID_STRIDE;
        let tables = self.tables.read().unwrap();
        let order = self.order.read().unwrap();
        // Fast path: the capacity bitmap points at the lowest-placed arena with a
        // free slot (find-first-set = lowest-first packing), skipping the
        // full-arena prefix that made this an O(num_arenas) walk per alloc at
        // pressure. A set bit can be stale (the arena filled since it was set) →
        // the claim fails → clear it and try the next set bit.
        while let Some(place) = self.capacity.first_set() {
            match Self::at_place(&tables, &order, place) {
                Some((arena_idx, table)) => {
                    if let Some(chunk_idx) = table.try_claim_one() {
                        if table.is_full() {
                            self.capacity.clear(place);
                        }
                        return Some(((arena_idx * stride + chunk_idx) as i64, Arc::clone(table)));
                    }
                    // Stale set bit — arena is full. Clear and try the next.
                    self.capacity.clear(place);
                }
                // Bit set for a tombstoned arena (register/release race), or left at
                // the place an arena moved from when it was ranked — clear.
                None => self.capacity.clear(place),
            }
        }
        // Fallback: the bitmap says every arena is full. That's authoritative
        // *unless* a bit was over-cleared (an alloc's fill-clear raced a dec's
        // set), which can only hide capacity, never invent fullness. Rebuild the
        // whole bitmap once from the authoritative `free_count` — recovering
        // every over-cleared bit in a single pass — then retry the fast path. We
        // hold `alloc_gate`, so no concurrent claim can consume the recovered
        // capacity before we do; concurrent drops only add more.
        let mut recovered = false;
        for table in tables.values() {
            if table.free_count() > 0 {
                self.capacity.set(table.place());
                recovered = true;
            }
        }
        if recovered {
            if let Some(place) = self.capacity.first_set() {
                if let Some((arena_idx, table)) = Self::at_place(&tables, &order, place) {
                    if let Some(chunk_idx) = table.try_claim_one() {
                        if table.is_full() {
                            self.capacity.clear(place);
                        }
                        return Some(((arena_idx * stride + chunk_idx) as i64, Arc::clone(table)));
                    }
                }
            }
        }
        None
    }

    /// Allocate up to `n` gids across this pool's arenas. Returns the
    /// pairs `(gid, table)` so the caller can construct `ChunkGid`s
    /// directly. May return fewer than `n` if the pool ran out of
    /// capacity; caller registers a fresh arena and retries.
    pub(super) fn allocate_n(&self, n: usize) -> Vec<(i64, Arc<ArenaRefcounts>)> {
        if n == 0 {
            return Vec::new();
        }
        let mut out: Vec<(i64, Arc<ArenaRefcounts>)> = Vec::with_capacity(n);

        // One gate acquisition for the whole bulk claim (see `allocate_any`).
        let _gate = self.alloc_gate.lock().unwrap();
        // The same lowest-placed-first order as `allocate_any`, draining each
        // arena before moving up — one read lock, no index collect + sort.
        let stride = GID_STRIDE;
        let tables = self.tables.read().unwrap();
        let order = self.order.read().unwrap();
        for arena_idx in order.values() {
            if out.len() == n {
                break;
            }
            let Some(table) = tables.get(arena_idx) else {
                continue;
            };
            let base = (*arena_idx * stride) as i64;
            // Drain as many as we can from this arena.
            while out.len() < n {
                match table.try_claim_one() {
                    Some(chunk_idx) => {
                        out.push((base + chunk_idx as i64, Arc::clone(table)));
                    }
                    None => break,
                }
            }
        }
        out
    }

    /// Allocate a gid from a specific arena. Returns `None` if the
    /// arena isn't registered with this pool, or has no free slots.
    pub(super) fn allocate_from_arena(
        &self,
        arena_idx: usize,
    ) -> Option<(i64, Arc<ArenaRefcounts>)> {
        // Gate with the other claim walks so `try_claim_one`'s relaxed probe
        // only ever races drops, never another claimer (see `allocate_any`).
        let _gate = self.alloc_gate.lock().unwrap();
        let table = {
            let tables = self.tables.read().unwrap();
            Arc::clone(tables.get(&arena_idx)?)
        };
        let chunk_idx = table.try_claim_one()?;
        let stride = GID_STRIDE;
        Some(((arena_idx * stride + chunk_idx) as i64, table))
    }

    /// Find a fully-free arena and tombstone it, returning its index. Skips
    /// arenas in `protected_arenas`. `None` when no arena is fully free.
    ///
    /// This used to refuse when releasing would leave under 10 % of an arena's
    /// slots free across the remaining pool, to stop steady-state churn —
    /// release an arena, immediately re-create it, each one a `cuMemAlloc` and
    /// a `Tensor::zeros` of 16 MiB. That guard's condition was "the pool is
    /// nearly full", which is exactly when reclaim was being asked for, so it
    /// needed a `force` bypass for the pressure path and the two disagreed
    /// about when reclaim was allowed at all.
    ///
    /// Under the reservation there is no churn to guard against: releasing is a
    /// push onto the free-region list and creating is a pop, both O(1) and
    /// neither touching the driver. So an empty arena is always released, and
    /// its region is immediately available to *any* class — §3.8's first
    /// pressure response, with nothing to decide.
    ///
    /// **A reserve class keeps one arena's worth free** (`reserve`): an empty
    /// arena is released only while the class would still hold that free
    /// without it.
    ///
    /// **The highest-placed empty goes first** — the one at the greatest offset
    /// into the reservation, never the greatest `arena_idx` — for every class: the
    /// arenas that stay are the lowest in the span, and the frontier comes down to
    /// them. Chosen by index, the kept reserve was wherever recycling had left the
    /// index: on BlackWorld (2026-11-08) an empty 2048 B reserve arena stood on
    /// region 3127, the top of the span, for hours, with 20 GB of holes below it
    /// that no pass could hand the expert zone.
    ///
    /// **A reserve does not hold the top while a hole lies below it.**
    /// `lowest_hole` is the lowest free region under the frontier, `None`
    /// where the pool has no regions. An empty arena a reserve would keep is
    /// released all the same when it stands above that hole: the class falls
    /// short of its reserve, the next gap owes it an arena, and a claim takes
    /// the lowest free region — the reserve moves down instead of pinning the
    /// frontier. On WhiteWorld (2026-11-10) a 1,024 B class kept an empty arena
    /// on region 487 as its reserve, with ~430 free regions (6.7 GB) below it
    /// and the expert zone 1 GB under its floor.
    pub(super) fn try_tombstone(
        &self,
        protected_arenas: &AHashSet<usize>,
        reserve: bool,
        lowest_hole: Option<usize>,
    ) -> Option<usize> {
        // Held across BOTH the emptiness test and the removal. Claims are gated
        // (`allocate_any` and friends take this before `tables.read()`), so with
        // it held no slot in the candidate can be occupied between observing
        // `live_count() == 0` and dropping the table. Without it, a claimer
        // slipping into that window leaves a live `ChunkGid` pointing into an
        // arena the caller then unmaps, and the freed index is recycled to
        // another format — cross-context KV contamination, not a clean fault.
        // Drops stay ungated: they can only take an arena from live to empty,
        // which never invalidates a decision made here.
        //
        // Lock order is `alloc_gate` → `tables`, matching every claim walk.
        // `next_tombstone` holds `metadata` outside this, and no claim path
        // takes `metadata`, so the two nest without a cycle.
        let _gate = self.alloc_gate.lock().unwrap();

        let tables = self.tables.read().unwrap();
        // Highest-placed fully-free, non-protected arena. `creation_pending` is
        // checked FIRST: its Acquire load pairs with `occupy`'s Release clear,
        // so an arena observed past its creation window is guaranteed to show
        // its first allocation in `live_count` — a freshly-registered arena
        // whose creator hasn't allocated yet can never be tombstoned (freeing
        // it would unmap memory an in-flight kernel writes, and recycle its
        // index to a second owner: cross-context KV contamination).
        let highest_empty = tables
            .iter()
            .filter(|(idx, t)| {
                !t.creation_pending() && t.live_count() == 0 && !protected_arenas.contains(idx)
            })
            .map(|(&idx, t)| (t.place(), idx))
            .max();
        if reserve {
            let free: usize = tables
                .values()
                .map(|t| self.arena_chunks.saturating_sub(t.live_count()))
                .sum();
            let above_a_hole = matches!(
                (highest_empty, lowest_hole),
                (Some((place, _)), Some(hole)) if place < RANKED_PLACES && hole < place
            );
            if free.saturating_sub(self.arena_chunks) < self.arena_chunks && !above_a_hole {
                return None;
            }
        }
        let (_, candidate) = highest_empty?;
        drop(tables);

        // Arena gone — its capacity bit goes with it, so `allocate_any` doesn't
        // chase a dangling place (it self-heals via the lookup miss anyway).
        self.remove_table(&mut self.tables.write().unwrap(), candidate);
        Some(candidate)
    }

    /// Remove `arena_idx`'s table if it is past its creation window and holds no live
    /// slot; `true` when it did.
    ///
    /// [`Self::try_tombstone`] for one named arena, under the same gate and for the same
    /// reason. Protection is not consulted: the only caller is the creator that holds
    /// the protection, releasing its own arena.
    pub(super) fn tombstone_if_empty(&self, arena_idx: usize) -> bool {
        let _gate = self.alloc_gate.lock().unwrap();
        let empty = self
            .tables
            .read()
            .unwrap()
            .get(&arena_idx)
            .is_some_and(|t| !t.creation_pending() && t.live_count() == 0);
        if !empty {
            return false;
        }
        self.remove_table(&mut self.tables.write().unwrap(), arena_idx)
    }

    /// Force-remove an arena's table regardless of whether it's empty.
    /// Used by the legacy `release_arena` path after a manual gid drain.
    pub(super) fn force_release(&self, arena_idx: usize) {
        self.remove_table(&mut self.tables.write().unwrap(), arena_idx);
    }

    pub(super) fn has_reclaimable(&self) -> bool {
        let tables = self.tables.read().unwrap();
        tables.values().any(|t| t.live_count() == 0)
    }

    /// Close `arena_idx`'s creation window. No-op if the arena is already gone
    /// or already past creation — see [`ArenaRefcounts::end_creation_window`].
    pub(super) fn finish_creation(&self, arena_idx: usize) {
        let tables = self.tables.read().unwrap();
        if let Some(t) = tables.get(&arena_idx) {
            t.end_creation_window();
        }
    }

    pub(super) fn free_count_for_arena(&self, arena_idx: usize) -> u32 {
        let tables = self.tables.read().unwrap();
        tables
            .get(&arena_idx)
            .map(|t| t.free_count() as u32)
            .unwrap_or(0)
    }

    /// Pool-wide live-slot count. O(1) via the shared atomic counter
    /// maintained by every [`ArenaRefcounts`] in this pool.
    #[inline]
    pub(super) fn total_live(&self) -> usize {
        self.total_live.load(Ordering::Relaxed)
    }

    /// Pool-wide free-slot count. O(1) via
    /// `total_arenas * arena_chunks - total_live`.
    #[inline]
    #[cfg(test)]
    pub(super) fn total_free(&self) -> usize {
        let arenas = self.total_arenas.load(Ordering::Relaxed);
        let total_slots = arenas.saturating_mul(self.arena_chunks);
        total_slots.saturating_sub(self.total_live())
    }

    /// Whole arenas this pool could free via perfect defragmentation:
    /// `total_arenas - ceil(total_live / arena_chunks)`. Zero means the pool is
    /// packed to within a single arena of free space, so a compaction pass
    /// would reclaim nothing — the signal to skip a futile (expensive) compact.
    pub(super) fn reclaimable_arenas(&self) -> usize {
        let arenas = self.total_arenas.load(Ordering::Relaxed);
        let live = self.total_live();
        let needed = if live == 0 {
            0
        } else {
            live.div_ceil(self.arena_chunks)
        };
        arenas.saturating_sub(needed)
    }

    /// Add this pool's arenas that hold no chunk to `census`, each counted once more
    /// under the reason [`Self::try_tombstone`] would pass over it, if it has one.
    pub(super) fn count_empty(&self, protected_arenas: &AHashSet<usize>, census: &mut EmptyCensus) {
        let tables = self.tables.read().unwrap();
        for (idx, t) in tables.iter() {
            if t.live_count() != 0 {
                continue;
            }
            census.empty += 1;
            if t.creation_pending() {
                census.in_creation += 1;
            }
            if protected_arenas.contains(idx) {
                census.protected += 1;
            }
        }
    }

    #[cfg(test)]
    pub(super) fn free_len(&self) -> usize {
        self.total_free()
    }
}
