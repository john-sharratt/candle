//! GID (Global Identifier) pool management for chunk allocation.
//!
//! Provides strongly-typed reference-counted GID allocation with automatic
//! return-to-pool semantics on drop. GIDs are partitioned by ArenaKey
//! (format + location) so allocations never land in wrong-format arenas.
//!
//! ## Design (lock-free refcount tables)
//!
//! Each registered arena has a single contiguous [`ArenaRefcounts`] struct
//! holding:
//!   - `counts: Vec<AtomicU16>` — one refcount per chunk slot. `0` means
//!     free; `≥ 1` means allocated with N holders.
//!   - `first_free: AtomicU64` — a best-effort hint pointing at the lowest
//!     known free slot. Allocators read it as a scan starting point; drops
//!     conditionally lower it via CAS when they free a slot below it.
//!
//! Every [`ChunkGid`] carries an `Arc<ArenaRefcounts>` (shared with every
//! other gid in the same arena) plus an `i64` id. No per-gid heap
//! allocation. Clone/drop are single atomic ops on `counts[chunk_idx]`.
//! At 10M-token arena scale, this replaces 10M `Arc<GidInner>` heap
//! allocations with one ~600 KB contiguous `Vec<AtomicU16>` per arena.
//!
//! ### Allocation path
//!
//! `ArenaPool::allocate_n` iterates arenas in `arena_idx` order. For each
//! arena, it scans `counts` forward from `first_free`, trying
//! `compare_exchange(0, 1)` on each slot. Success claims the slot;
//! failure means another thread got it first or it's occupied — scan
//! continues. After a claim, the allocator does a best-effort
//! `first_free.compare_exchange(start, i + 1)` to advance the hint.
//!
//! ### Drop path
//!
//! `ChunkGid::drop` does `counts[chunk_idx].fetch_sub(1)`. If the value
//! transitioned to zero, the slot is now free and a tiny CAS-loop tries
//! to lower `first_free` to `chunk_idx`. If the CAS loses to another
//! thread, the hint just stays higher than optimal — correctness is
//! unaffected; the next alloc will scan past stale entries and find the
//! actual free slot.
//!
//! No mutex anywhere on the hot path. The pool's `RwLock<HashMap>` is
//! taken only on `register_arena` / `release_arena` (rare).
//!
//! ## What was replaced
//!
//! The previous design held the free set as `Mutex<BinaryHeap<Reverse<i64>>>`
//! + `BTreeSet<arena_idx>` + `HashMap<arena_idx, free_count>` per format.
//! Every alloc and drop took the per-format mutex; every gid was a
//! separate `Arc<GidInner>` heap allocation. The lock-free refcount-table
//! design collapses all of that into one contiguous `Vec<AtomicU16>` per
//! arena and lets drops touch a single cache line.

use ahash::{AHashMap, AHashSet};
use candle::DType;
use std::collections::BTreeMap;
use std::{
    collections::VecDeque,
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU16, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, RwLock,
    },
};
use strum::IntoEnumIterator;

use super::arena::ArenaKey;
use crate::kv_cache::chunked::types::{arena_chunks_for_format, arena_gid_stride};
use crate::kv_cache::{ArenaLocation, KvFormat, QuantFormat};

/// Per-arena refcount table. Lives behind an `Arc` shared by every
/// `ChunkGid` allocated from this arena. Lock-free: all mutation is via
/// atomics on `counts[chunk_idx]` and `first_free`.
#[derive(Debug)]
pub struct ArenaRefcounts {
    /// Refcount per chunk slot. `0` = free, `≥ 1` = allocated. This is the
    /// **authority**; the `compare_exchange(0→1)` against it is the real
    /// claim.
    counts: Vec<AtomicU16>,
    /// Occupancy summary: one BIT per slot (set ⟺ `counts[i] > 0`), packed
    /// 64 slots per `u64` word, so the scan skips 64 occupied slots in one
    /// relaxed load instead of probing each `counts` entry.
    ///
    /// This is a **scan hint only** — never consulted for correctness. It is
    /// maintained on exactly the same `0↔1` transitions as `live`: the bit is
    /// set right after a successful claim in `try_claim_one`, and cleared on
    /// the `1→0` last drop in `dec`. `inc`/non-last `dec` (clone / shared
    /// drop) never cross `0↔1` and never touch it. Updates use atomic
    /// `fetch_or` / `fetch_and` so concurrent set/clear of *different* bits in
    /// the same word compose without lost updates. See the race analysis
    /// above `try_claim_one`.
    occupancy: Vec<AtomicU64>,
    /// Best-effort hint for the lowest known free slot. Allocators read
    /// it as a scan starting point; drops lower it via CAS when they
    /// free a slot below it.
    first_free: AtomicU64,
    /// Number of currently-allocated slots in this arena (slots with
    /// `counts[i] > 0`). Maintained on each successful alloc/drop
    /// transition. Used by `try_tombstone` (arena is tombstoneable
    /// when `live == 0`) and diagnostics.
    live: AtomicUsize,
    /// Pool-wide live counter, shared with [`ArenaPool::total_live`]
    /// and every other arena in the same per-format pool. Updated on
    /// the same `0 → 1` / `1 → 0` transitions as `live`, so
    /// `ArenaPool::total_live()` is an O(1) atomic load instead of a
    /// per-call sum across arenas.
    pool_total_live: Arc<AtomicUsize>,
    /// Total chunk capacity of this arena (constant for the arena's
    /// lifetime, set at `register_arena` time).
    arena_chunks: usize,
    /// Global arena index — encoded into every gid as the upper bits.
    arena_idx: usize,
    /// Format key this arena was registered with. Inlined here so
    /// `ChunkGid::route_key` is a single pointer-deref away.
    key: ArenaKey,
}

impl ArenaRefcounts {
    fn new(
        arena_chunks: usize,
        arena_idx: usize,
        key: ArenaKey,
        pool_total_live: Arc<AtomicUsize>,
    ) -> Self {
        let mut counts = Vec::with_capacity(arena_chunks);
        for _ in 0..arena_chunks {
            counts.push(AtomicU16::new(0));
        }
        // One bit per slot, 64 per word, rounded up. All zero ⇒ all free,
        // matching the all-zero `counts` initial state.
        let n_words = arena_chunks.div_ceil(64);
        let mut occupancy = Vec::with_capacity(n_words);
        for _ in 0..n_words {
            occupancy.push(AtomicU64::new(0));
        }
        Self {
            counts,
            occupancy,
            first_free: AtomicU64::new(0),
            live: AtomicUsize::new(0),
            pool_total_live,
            arena_chunks,
            arena_idx,
            key,
        }
    }

    /// Mark slot `i` occupied in the scan bitmap (set its bit).
    ///
    /// Atomic `fetch_or` so it composes with a concurrent `set_free` on a
    /// *different* bit of the same word without a lost update. Relaxed is
    /// sufficient: this is only read by the (gated) scan, and cross-claimer
    /// visibility is carried by `ArenaPool::alloc_gate`'s lock release/acquire;
    /// same-thread reuse is sequenced.
    #[inline]
    fn set_occupied(&self, i: usize) {
        self.occupancy[i / 64].fetch_or(1u64 << (i % 64), Ordering::Relaxed);
    }

    /// Mark slot `i` free in the scan bitmap (clear its bit).
    ///
    /// Atomic `fetch_and` so it composes with a concurrent `set_occupied`.
    /// **Release**: `dec` calls this and then lowers `first_free` with a
    /// Release CAS, and the scan reads bitmap words with Acquire — so a
    /// claimer that observes either the cleared bit or the lowered hint is
    /// guaranteed to see this clear (drops are not gated, so this is the
    /// synchronization that publishes a free slot to the scanner).
    #[inline]
    fn set_free(&self, i: usize) {
        self.occupancy[i / 64].fetch_and(!(1u64 << (i % 64)), Ordering::Release);
    }

    // ── Occupancy-bitmap race analysis ───────────────────────────────────────
    //
    // `counts[i]` is the truth; the bitmap is a hint and the `compare_exchange`
    // below is the authoritative claim. Two skews are possible; neither is a
    // correctness bug:
    //
    //  * **bit set while slot free (would leak the slot): IMPOSSIBLE as a
    //    persistent state.** A bit is only set by `set_occupied`, always right
    //    after a successful `cmpxchg(0→1)` (i.e. with `count == 1`). For a
    //    slot to reach `count == 0` a last-drop ran, and that `dec` always
    //    calls `set_free` (an atomic `fetch_and`, never lost). Any `set_occupied`
    //    after that implies another `cmpxchg(0→1)` → `count == 1` again. So
    //    whenever `count` is stably 0 the bit is 0. The only `bit=1, count=0`
    //    instant is the window *inside* a single `dec` between its `fetch_sub`
    //    and its `set_free`, which the same thread closes immediately.
    //
    //  * **bit clear while slot occupied (a wasted probe): possible, transient,
    //    self-healing.** If a `dec` frees a slot (`fetch_sub`→0) and a claimer
    //    re-takes it (`cmpxchg 0→1`, `set_occupied`) before the `dec`'s trailing
    //    `set_free` lands, the late `set_free` can clear a bit that now belongs
    //    to the new owner → `bit=0, count>0`. The scan then probes that slot and
    //    the `cmpxchg` cleanly fails (count≠0) — one wasted relaxed read+cmpxchg.
    //    It heals on that owner's next drop (`count→0` with the bit already 0 =
    //    consistent free). We deliberately do NOT re-set the bit on a failed
    //    claim: a concurrent `dec` could be freeing the slot, and setting it
    //    then would manufacture the leak skew above.
    //
    // Same-word concurrency is safe because `set_occupied`/`set_free` are atomic
    // RMWs (OR / AND), so two threads flipping different bits of one word both
    // take effect.

    /// Try to claim a single free slot at or after the `first_free` hint.
    /// Returns the chunk_idx claimed, or `None` if no free slot is observable.
    ///
    /// Callers serialize this via `ArenaPool::alloc_gate`, so at most one
    /// claimer scans at a time; concurrent `dec` drops (ungated) only ever
    /// *free* slots and clear bits.
    fn try_claim_one(&self) -> Option<usize> {
        let n_words = self.occupancy.len();
        let mut start = self.first_free.load(Ordering::Acquire) as usize;
        loop {
            let mut w = start / 64;
            // In the first scanned word, ignore bits below `start`.
            let mut first_word_skip = (start % 64) as u32;
            while w < n_words {
                // Free bits = zero bits of the occupancy word. Acquire pairs
                // with the Release `set_free` in `dec` so a freed slot is
                // visible here.
                let occ = self.occupancy[w].load(Ordering::Acquire);
                let mut free = !occ;
                if first_word_skip != 0 {
                    free &= u64::MAX << first_word_skip;
                    first_word_skip = 0;
                }
                while free != 0 {
                    let b = free.trailing_zeros() as usize;
                    let i = w * 64 + b;
                    if i >= self.arena_chunks {
                        // Phantom high bits past capacity in the final word —
                        // always zero in `occupancy`, never real free slots.
                        break;
                    }
                    // Authoritative claim — the bitmap is only a hint.
                    if self.counts[i]
                        .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                    {
                        self.set_occupied(i);
                        let _ = self.first_free.compare_exchange(
                            start as u64,
                            (i + 1) as u64,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        );
                        self.live.fetch_add(1, Ordering::Relaxed);
                        self.pool_total_live.fetch_add(1, Ordering::Relaxed);
                        return Some(i);
                    }
                    // Stale-free bit (occupied despite the 0): skip it. Do not
                    // set it — see the race analysis above.
                    free &= free - 1;
                }
                w += 1;
            }
            // Scanned [start, arena_chunks) with nothing claimable. If a
            // concurrent drop lowered `first_free` below `start`, rescan that
            // lower window; otherwise the arena is (observably) full.
            let new_start = self.first_free.load(Ordering::Acquire) as usize;
            if new_start >= start {
                return None;
            }
            start = new_start;
        }
    }

    /// Increment a slot's refcount — called by `ChunkGid::clone`. The
    /// slot must already be allocated (count ≥ 1); we never need
    /// Acquire semantics because the data is already visible.
    #[inline]
    fn inc(&self, chunk_idx: usize) {
        self.counts[chunk_idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement a slot's refcount — called by `ChunkGid::drop`. If the
    /// count transitions to 0, lower `first_free` if this slot is below
    /// it (best-effort CAS loop), and decrement the live counters
    /// (per-arena and pool-wide).
    #[inline]
    fn dec(&self, chunk_idx: usize) {
        let prev = self.counts[chunk_idx].fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.live.fetch_sub(1, Ordering::Relaxed);
            self.pool_total_live.fetch_sub(1, Ordering::Relaxed);
            // Clear the scan bitmap bit (Release) BEFORE lowering `first_free`
            // below, so a claimer that Acquire-observes the lowered hint also
            // sees this freed slot's bit. Sequenced before the first_free CAS.
            self.set_free(chunk_idx);
            let mut cur = self.first_free.load(Ordering::Acquire);
            while (chunk_idx as u64) < cur {
                match self.first_free.compare_exchange_weak(
                    cur,
                    chunk_idx as u64,
                    Ordering::Release,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(actual) => cur = actual,
                }
            }
        } else if prev == 0 {
            // Underflow: someone over-decremented. Panic loudly — this
            // is a logic bug (mismatched Clone/Drop), not something the
            // pool can recover from.
            panic!(
                "ArenaRefcounts::dec: refcount underflow at arena {} chunk {}",
                self.arena_idx, chunk_idx
            );
        }
    }

    /// Read a slot's current refcount. Used by [`ChunkGid::strong_count`]
    /// and consumers that need COW detection.
    #[inline]
    fn load(&self, chunk_idx: usize) -> u16 {
        self.counts[chunk_idx].load(Ordering::Relaxed)
    }

    /// Number of currently-allocated slots in this arena.
    #[inline]
    fn live_count(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    /// Number of currently-free slots in this arena.
    #[inline]
    fn free_count(&self) -> usize {
        self.arena_chunks.saturating_sub(self.live_count())
    }

    /// Iterate over the raw gids currently allocated in this arena.
    /// Slow path used by diagnostics / live-gid queries — scans every
    /// slot. Only called from the CUDA-gated defrag path.
    #[cfg(feature = "cuda")]
    fn live_gids(&self) -> Vec<i64> {
        let base = (self.arena_idx * arena_gid_stride()) as i64;
        let mut out = Vec::with_capacity(self.live_count());
        for i in 0..self.arena_chunks {
            if self.counts[i].load(Ordering::Relaxed) > 0 {
                out.push(base + i as i64);
            }
        }
        out
    }
}

/// Refcount backing for a [`ChunkGid`].
///
/// - `Pooled`: real arena gid — refcount lives in the arena's shared
///   [`ArenaRefcounts`] table (no per-gid heap allocation).
/// - `Detached`: sentinel / test gid not backed by any production pool
///   — refcount is per-gid, in a dedicated `Arc<AtomicU16>`. This
///   variant *does* heap-allocate per `::detached()` call, but those
///   calls are rare (error paths, resume codecs, test fixtures).
#[derive(Debug)]
enum GidBacking {
    Pooled(Arc<ArenaRefcounts>),
    Detached(Arc<AtomicU16>),
}

/// Strongly-typed chunk global identifier with automatic pooling.
///
/// Holds an `Arc<ArenaRefcounts>` pointing to its arena's refcount
/// table (for pooled gids) or a per-gid `Arc<AtomicU16>` (for
/// detached/sentinel gids). `Clone` increments the slot's refcount;
/// `Drop` decrements and recycles when the count reaches zero. No
/// per-gid heap allocation for pooled gids — the `Arc` is shared with
/// every other `ChunkGid` allocated from the same arena.
pub struct ChunkGid {
    id: i64,
    backing: GidBacking,
}

impl ChunkGid {
    /// Create a detached Gid not backed by any live allocation pool.
    ///
    /// Used for sentinel handles (e.g. `id = -1`) in error paths,
    /// resume codecs, and tests. Detached gids carry their own
    /// per-gid refcount, so `Clone` / `strong_count` / `is_shared`
    /// still work — but `Drop` doesn't touch any production pool.
    pub fn detached(id: i64) -> Self {
        Self {
            id,
            backing: GidBacking::Detached(Arc::new(AtomicU16::new(1))),
        }
    }

    /// Refcount of this gid (logical, not Arc-related).
    ///
    /// Used for COW detection: a count of 1 means exclusively owned;
    /// `> 1` means shared (another `ChunkRef` / `SealedChunk` / view
    /// also holds a clone).
    pub fn strong_count(&self) -> usize {
        match &self.backing {
            GidBacking::Pooled(t) => {
                let chunk_idx = (self.id as usize) % arena_gid_stride();
                t.load(chunk_idx) as usize
            }
            GidBacking::Detached(c) => c.load(Ordering::Relaxed) as usize,
        }
    }

    /// `true` when another holder shares this gid (COW needed on write).
    pub fn is_shared(&self) -> bool {
        self.strong_count() > 1
    }

    /// `true` when this is the sole owner.
    pub fn is_unique(&self) -> bool {
        self.strong_count() == 1
    }

    /// Extract the raw i64 value.
    #[inline]
    pub fn raw(&self) -> i64 {
        self.id
    }

    /// `true` when this is the empty/sentinel slot (raw id == -1).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id == -1
    }

    /// Arena index this gid belongs to.
    #[inline]
    pub fn arena_idx(&self) -> usize {
        self.id as usize / arena_gid_stride()
    }

    /// Chunk offset within its arena.
    #[inline]
    pub fn chunk_idx(&self) -> usize {
        self.id as usize % arena_gid_stride()
    }

    /// Arena routing key (format + location) this gid was allocated
    /// from. Returns `None` for detached/sentinel gids.
    pub fn route_key(&self) -> Option<&ArenaKey> {
        match &self.backing {
            GidBacking::Pooled(t) => Some(&t.key),
            GidBacking::Detached(_) => None,
        }
    }
}

impl Clone for ChunkGid {
    fn clone(&self) -> Self {
        match &self.backing {
            GidBacking::Pooled(t) => {
                let chunk_idx = (self.id as usize) % arena_gid_stride();
                t.inc(chunk_idx);
                Self {
                    id: self.id,
                    backing: GidBacking::Pooled(Arc::clone(t)),
                }
            }
            GidBacking::Detached(c) => {
                c.fetch_add(1, Ordering::Relaxed);
                Self {
                    id: self.id,
                    backing: GidBacking::Detached(Arc::clone(c)),
                }
            }
        }
    }
}

impl Drop for ChunkGid {
    fn drop(&mut self) {
        match &self.backing {
            GidBacking::Pooled(t) => {
                let chunk_idx = (self.id as usize) % arena_gid_stride();
                t.dec(chunk_idx);
            }
            GidBacking::Detached(c) => {
                // Per-gid refcount; we never recycle into a pool, so
                // the only side-effect is the decrement itself (the
                // `Arc<AtomicU16>` is dropped naturally when its last
                // reference goes).
                c.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }
}

impl fmt::Debug for ChunkGid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkGid").field("id", &self.id).finish()
    }
}

impl PartialEq for ChunkGid {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ChunkGid {}

/// Per-format arena pool: refcount tables keyed by arena_idx.
///
/// Each arena owns a lock-free [`ArenaRefcounts`] table behind an `Arc`.
/// The pool only takes its `RwLock` on `register_arena` / `release_arena`
/// (rare); allocation walks the tables read-locked and operates lock-
/// free against the chosen arena's counts.
#[derive(Debug)]
struct ArenaPool {
    /// `arena_idx → refcount table`. `RwLock`'d for register/release;
    /// reads are uncontended in steady state.
    /// Keyed by `arena_idx`. A `BTreeMap` (not a hash map) so iteration is
    /// already in ascending `arena_idx` order: the allocators walk it
    /// lowest-first for compaction-friendly packing without rebuilding and
    /// sorting an index `Vec` on every allocation.
    tables: RwLock<BTreeMap<usize, Arc<ArenaRefcounts>>>,
    /// Number of currently-registered arenas for this key. Lock-free
    /// counter for fast diagnostics.
    total_arenas: AtomicUsize,
    /// Running count of currently-allocated slots across every arena
    /// in this per-format pool. Updated by [`ArenaRefcounts`] on each
    /// `0 → 1` / `1 → 0` transition, so `total_live()` is an O(1)
    /// atomic load. Shared (`Arc`) with every refcount table.
    total_live: Arc<AtomicUsize>,
    /// Physical chunk capacity for one arena of this specific format.
    arena_chunks: usize,
    /// Serializes the *claiming* walk (`allocate_any` / `allocate_n` /
    /// `allocate_excluding`). Only one thread scans this pool's `counts`
    /// arrays for a free slot at a time, so the scan reads a slot's
    /// occupancy with a plain relaxed load (no per-slot locked RMW) and the
    /// 128 worker/persist threads stop ping-ponging the same cache lines.
    /// Drops (`ArenaRefcounts::dec`) are deliberately NOT gated — they run
    /// lock-free on arbitrary threads; the gate only removes claimer↔claimer
    /// contention. Held for bookkeeping only, never across GPU work.
    alloc_gate: Mutex<()>,
}

impl ArenaPool {
    fn new(format: KvFormat) -> Self {
        Self {
            tables: RwLock::new(BTreeMap::new()),
            total_arenas: AtomicUsize::new(0),
            total_live: Arc::new(AtomicUsize::new(0)),
            arena_chunks: arena_chunks_for_format(format),
            alloc_gate: Mutex::new(()),
        }
    }

    /// Register a new arena with the pool — creates its refcount table.
    fn register_arena(&self, arena_idx: usize, key: ArenaKey) -> Arc<ArenaRefcounts> {
        let table = Arc::new(ArenaRefcounts::new(
            self.arena_chunks,
            arena_idx,
            key,
            Arc::clone(&self.total_live),
        ));
        {
            let mut tables = self.tables.write().unwrap();
            tables.insert(arena_idx, Arc::clone(&table));
        }
        self.total_arenas.fetch_add(1, Ordering::Relaxed);
        table
    }

    /// Allocate a single gid from any arena. Iterates arenas in
    /// `arena_idx` order (lowest first) so live data clusters in low
    /// indices, keeping compaction effective.
    fn allocate_any(&self) -> Option<(i64, Arc<ArenaRefcounts>)> {
        // Serialize the claiming walk: only one thread scans this pool's
        // `counts` arrays at a time, so `try_claim_one` can probe occupancy
        // with a relaxed load (no per-slot locked RMW) without 128 threads
        // ping-ponging the same cache lines. Drops stay lock-free.
        let _gate = self.alloc_gate.lock().unwrap();
        // `tables` is a BTreeMap, so `iter()` yields arenas in ascending
        // arena_idx order — the lowest-first packing that keeps live data
        // clustered for compaction — with no per-call index collect + sort.
        let stride = arena_gid_stride();
        let tables = self.tables.read().unwrap();
        for (&arena_idx, table) in tables.iter() {
            if let Some(chunk_idx) = table.try_claim_one() {
                let gid = (arena_idx * stride + chunk_idx) as i64;
                return Some((gid, Arc::clone(table)));
            }
        }
        None
    }

    /// Allocate up to `n` gids across this pool's arenas. Returns the
    /// pairs `(gid, table)` so the caller can construct `ChunkGid`s
    /// directly. May return fewer than `n` if the pool ran out of
    /// capacity; caller registers a fresh arena and retries.
    fn allocate_n(&self, n: usize) -> Vec<(i64, Arc<ArenaRefcounts>)> {
        if n == 0 {
            return Vec::new();
        }
        let mut out: Vec<(i64, Arc<ArenaRefcounts>)> = Vec::with_capacity(n);

        // One gate acquisition for the whole bulk claim (see `allocate_any`).
        let _gate = self.alloc_gate.lock().unwrap();
        // Same lowest-first BTreeMap walk as `allocate_any`, draining each
        // arena before moving up — one read lock, no index collect + sort.
        let stride = arena_gid_stride();
        let tables = self.tables.read().unwrap();
        for (&arena_idx, table) in tables.iter() {
            if out.len() == n {
                break;
            }
            let base = (arena_idx * stride) as i64;
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
    fn allocate_from_arena(&self, arena_idx: usize) -> Option<(i64, Arc<ArenaRefcounts>)> {
        // Gate with the other claim walks so `try_claim_one`'s relaxed probe
        // only ever races drops, never another claimer (see `allocate_any`).
        let _gate = self.alloc_gate.lock().unwrap();
        let table = {
            let tables = self.tables.read().unwrap();
            Arc::clone(tables.get(&arena_idx)?)
        };
        let chunk_idx = table.try_claim_one()?;
        let stride = arena_gid_stride();
        Some(((arena_idx * stride + chunk_idx) as i64, table))
    }

    /// Allocate a gid from any arena **except** `exclude_arena`. Used
    /// during greedy arena drain so destination slots never land in
    /// the arena being evicted. CUDA-only — the defrag path that
    /// uses it is gated behind the cuda feature.
    #[cfg(feature = "cuda")]
    fn allocate_excluding(&self, exclude_arena: usize) -> Option<(i64, Arc<ArenaRefcounts>)> {
        // Serialize with the other claim walks (see `allocate_any`).
        let _gate = self.alloc_gate.lock().unwrap();
        let stride = arena_gid_stride();
        let tables = self.tables.read().unwrap();
        for (&arena_idx, table) in tables.iter() {
            if arena_idx == exclude_arena {
                continue;
            }
            if let Some(chunk_idx) = table.try_claim_one() {
                return Some(((arena_idx * stride + chunk_idx) as i64, Arc::clone(table)));
            }
        }
        None
    }

    /// Find a fully-free arena and tombstone it. Returns the arena's
    /// index. Skips arenas in `protected_arenas`.
    ///
    /// Returns `None` when:
    ///   - No arena is fully free, OR
    ///   - Releasing would leave < 10% of `arena_chunks` of headroom
    ///     across the remaining pool (matches prior behaviour to avoid
    ///     thrashing).
    fn try_tombstone(&self, protected_arenas: &AHashSet<usize>) -> Option<usize> {
        // Free-headroom check first — derived in O(1) from the running
        // counters. After releasing one arena we'd have
        // `(total_arenas - 1) * arena_chunks - total_live` free slots;
        // bail if that's under the 10% thrash threshold without even
        // taking the tables read lock.
        let arenas = self.total_arenas.load(Ordering::Relaxed);
        if arenas == 0 {
            return None;
        }
        let total_slots_after = arenas.saturating_sub(1).saturating_mul(self.arena_chunks);
        let after_release = total_slots_after.saturating_sub(self.total_live());
        if after_release < self.arena_chunks / 10 {
            return None;
        }

        let tables = self.tables.read().unwrap();
        // Lowest-index fully-free, non-protected arena.
        let candidate = tables
            .iter()
            .filter(|(idx, t)| t.live_count() == 0 && !protected_arenas.contains(idx))
            .map(|(&idx, _)| idx)
            .min()?;
        drop(tables);

        let mut tables = self.tables.write().unwrap();
        tables.remove(&candidate);
        self.total_arenas.fetch_sub(1, Ordering::Relaxed);
        Some(candidate)
    }

    /// Force-remove an arena's table regardless of whether it's empty.
    /// Used by the legacy `release_arena` path after a manual gid drain.
    fn force_release(&self, arena_idx: usize) {
        let mut tables = self.tables.write().unwrap();
        if tables.remove(&arena_idx).is_some() {
            self.total_arenas.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn has_reclaimable(&self) -> bool {
        let tables = self.tables.read().unwrap();
        tables.values().any(|t| t.live_count() == 0)
    }

    fn free_count_for_arena(&self, arena_idx: usize) -> u32 {
        let tables = self.tables.read().unwrap();
        tables
            .get(&arena_idx)
            .map(|t| t.free_count() as u32)
            .unwrap_or(0)
    }

    /// Pool-wide live-slot count. O(1) via the shared atomic counter
    /// maintained by every [`ArenaRefcounts`] in this pool.
    #[inline]
    fn total_live(&self) -> usize {
        self.total_live.load(Ordering::Relaxed)
    }

    /// Pool-wide free-slot count. O(1) via
    /// `total_arenas * arena_chunks - total_live`.
    #[inline]
    #[cfg(test)]
    fn total_free(&self) -> usize {
        let arenas = self.total_arenas.load(Ordering::Relaxed);
        let total_slots = arenas.saturating_mul(self.arena_chunks);
        total_slots.saturating_sub(self.total_live())
    }

    fn defragmentable_ratio(&self) -> f32 {
        let arenas = self.total_arenas.load(Ordering::Relaxed);
        if arenas == 0 {
            return 0.0;
        }
        let live = self.total_live();
        let needed = if live == 0 {
            0
        } else {
            live.div_ceil(self.arena_chunks)
        };
        (arenas.saturating_sub(needed)) as f32 / arenas as f32
    }

    /// Whole arenas this pool could free via perfect defragmentation:
    /// `total_arenas - ceil(total_live / arena_chunks)`. Zero means the pool is
    /// packed to within a single arena of free space, so a compaction pass
    /// would reclaim nothing — the signal to skip a futile (expensive) compact.
    fn reclaimable_arenas(&self) -> usize {
        let arenas = self.total_arenas.load(Ordering::Relaxed);
        let live = self.total_live();
        let needed = if live == 0 {
            0
        } else {
            live.div_ceil(self.arena_chunks)
        };
        arenas.saturating_sub(needed)
    }

    /// CUDA-only: list every live gid in a specific arena. Used by the
    /// defrag/eviction path to remap chunks before tombstoning the
    /// drained arena.
    #[cfg(feature = "cuda")]
    fn live_gids_for_arena(&self, arena_idx: usize) -> Vec<i64> {
        let tables = self.tables.read().unwrap();
        tables
            .get(&arena_idx)
            .map(|t| t.live_gids())
            .unwrap_or_default()
    }

    /// CUDA-only: arenas sorted by live-slot count (emptiest first) for
    /// the defrag/eviction pass to pick drain targets.
    #[cfg(feature = "cuda")]
    fn arenas_sorted_by_live(&self) -> Vec<(usize, usize)> {
        let tables = self.tables.read().unwrap();
        let mut arenas: Vec<(usize, usize)> = tables
            .iter()
            .map(|(&idx, t)| (idx, t.live_count()))
            .collect();
        arenas.sort_by_key(|&(_, live)| live);
        arenas
    }

    #[cfg(test)]
    fn free_len(&self) -> usize {
        self.total_free()
    }
}

/// Internal state of the GID pool, partitioned by ArenaKey.
struct GidPoolState {
    /// arena_idx → ArenaKey registry for routing/compaction bookkeeping.
    /// None = tombstoned (index recycled) or not yet assigned.
    arena_registry: Vec<Option<ArenaKey>>,
    /// Monotonic arena index allocator (fallback when `free_arenas` is empty).
    next_arena_idx: usize,
    /// FIFO queue of recycled arena indices from tombstoned arenas.
    free_arenas: VecDeque<usize>,
    /// Arena indices pinned for the lifetime of this backing.
    /// Protected arenas are never tombstoned by compaction.
    protected_arenas: AHashSet<usize>,
}

/// Inner pool state.
struct GidPoolInner {
    /// Immutable key → per-format pool table. Each pool has its own
    /// `RwLock<HashMap>` for register/release; alloc is lock-free.
    pools: AHashMap<ArenaKey, ArenaPool>,
    /// Mutable metadata for arena routing/recycling.
    metadata: Mutex<GidPoolState>,
    /// Lock-free hint for reclaim detection.
    /// `false` = definitely no fully-free arena.
    /// `true`  = maybe reclaimable; verify under lock.
    may_have_reclaimable: AtomicBool,
}

fn preallocated_pool_table() -> AHashMap<ArenaKey, ArenaPool> {
    // Locations × (DType variants + QuantFormat variants). With current
    // enums this is ~58 entries; 64 leaves room without rehashing on
    // first registration.
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

/// Pool for allocating and recycling gids, partitioned by ArenaKey.
pub struct ChunkGidPool {
    inner: Arc<GidPoolInner>,
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
            state.arena_registry[arena_idx] = Some(key.clone());
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
            state.arena_registry[arena_idx] = Some(key.clone());
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

    /// Allocate a gid for the given format.
    ///
    /// Iterates arenas in `arena_idx` order (lowest first). Returns
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

    /// Allocate a gid for `key` from any arena **except**
    /// `exclude_arena`. Used during greedy arena eviction.
    #[cfg(feature = "cuda")]
    pub fn allocate_for_excluding(&self, key: ArenaKey, exclude_arena: usize) -> Option<ChunkGid> {
        let pool = self.inner.pools.get(&key)?;
        let (id, table) = pool.allocate_excluding(exclude_arena)?;
        Some(ChunkGid {
            id,
            backing: GidBacking::Pooled(table),
        })
    }

    /// Return the live gids for `arena_idx` (slots with refcount > 0).
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
    #[cfg(feature = "cuda")]
    pub fn arenas_sorted_by_live_for_key(&self, key: &ArenaKey) -> Vec<(usize, usize)> {
        self.inner
            .pools
            .get(key)
            .map(|pool| pool.arenas_sorted_by_live())
            .unwrap_or_default()
    }

    /// Convenience: allocate a gid using a default test key.
    pub fn allocate(&self) -> ChunkGid {
        let key = ArenaKey::gpu_float(candle::DType::BF16);
        if let Some(gid) = self.allocate_for(key.clone()) {
            return gid;
        }
        self.register_arena(key.clone());
        self.allocate_for(key)
            .expect("just registered arena, must have capacity")
    }

    /// Find a fully-free arena of this format and release it.
    pub fn next_tombstone(&self, key: ArenaKey) -> Option<usize> {
        let mut state = self.inner.metadata.lock().unwrap();
        let pool = self.inner.pools.get(&key)?;
        let arena_idx = pool.try_tombstone(&state.protected_arenas)?;
        state.arena_registry[arena_idx] = None;
        state.free_arenas.push_back(arena_idx);
        Some(arena_idx)
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

    /// Force-release an arena's bookkeeping after an external drain
    /// of its gids. Used by the compaction path that manually frees
    /// chunks then expects the pool slot to disappear.
    pub fn force_release_arena(&self, arena_idx: usize) {
        let key = {
            let mut state = self.inner.metadata.lock().unwrap();
            let key = state.arena_registry.get(arena_idx).and_then(|k| k.clone());
            if key.is_some() {
                state.arena_registry[arena_idx] = None;
                state.free_arenas.push_back(arena_idx);
            }
            key
        };
        if let Some(key) = key {
            if let Some(pool) = self.inner.pools.get(&key) {
                pool.force_release(arena_idx);
            }
        }
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

    /// Lock-free hint: return true if any key can free at least
    /// `threshold` of its currently registered arenas via perfect
    /// defragmentation.
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

    /// Maximum gid currently in circulation (for compaction bound checks).
    pub fn max_gid(&self) -> Option<i64> {
        let state = self.inner.metadata.lock().ok()?;
        let mut max: i64 = -1;
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

    /// Resync diagnostic counters from authoritative table state.
    /// With the new lock-free design the counters are inherently
    /// up-to-date — this method is a no-op kept for API compatibility.
    pub(crate) fn resync_counters(&self) {}

    /// Number of free gids for a specific arena index. Returns 0 if
    /// the arena isn't registered or has no free gids.
    pub(crate) fn arena_free_count(&self, arena_idx: usize) -> u32 {
        let state = self.inner.metadata.lock().unwrap();
        let key = match state.arena_registry.get(arena_idx) {
            Some(Some(k)) => k.clone(),
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

    #[test]
    fn test_register_and_allocate() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let arena_idx = pool.register_arena(key.clone());
        assert_eq!(arena_idx, 0);

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

        // After drop, the slot is free again — total free == capacity.
        assert_eq!(pool.free_list_len_for(key.clone()), test_arena_chunks());

        // Allocating again should reuse it (lowest gid via first_free hint).
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

        // Drop original — clone still holds a refcount.
        drop(gid1);
        assert_eq!(pool.free_list_len_for(key.clone()), free_after_alloc);

        // Drop clone — now freed.
        drop(gid1_clone);
        assert_eq!(pool.free_list_len_for(key.clone()), free_after_alloc + 1);

        let gid2 = pool.allocate_for(key.clone()).unwrap();
        assert_eq!(gid2.raw(), gid1_raw);
    }

    #[test]
    fn test_strong_count_tracks_clones() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key.clone());
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
        let a = pool.register_arena(key.clone());
        let b = pool.register_arena(key.clone());
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        // Each arena contributes `arena_chunks` of capacity.
        assert_eq!(pool.free_list_len_for(key), test_arena_chunks() * 2);
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

        // 2 lives, 2 arenas, capacity per arena `arena_chunks`. live ÷
        // arena_chunks rounded up = 1 arena needed; 1 of 2 reclaimable.
        assert!(pool.defragmentable_ratio_for(&key) > 0.0);
    }

    #[test]
    fn can_reclaim_arena_only_when_a_whole_arena_is_recoverable() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        // One arena holding a live chunk: less than a whole arena of free space,
        // so a forced compaction can release nothing.
        pool.register_arena(key.clone());
        let _a0 = pool.allocate_from_arena(key.clone(), 0).unwrap();
        assert!(
            !pool.can_reclaim_arena(),
            "1 arena with a live chunk: nothing whole to reclaim"
        );
        // Add a second, empty arena: a whole arena's worth of free space is now
        // recoverable (needed = ceil(1 live / arena_chunks) = 1, of 2 arenas).
        pool.register_arena(key.clone());
        assert!(
            pool.can_reclaim_arena(),
            "2 arenas, 1 live chunk: one whole arena is reclaimable"
        );
    }
}
