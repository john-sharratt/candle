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
//!   - `counts: Vec<AtomicU16>` — one word per chunk slot, overlapped by the
//!     `occupancy` bit: the slot's **refcount** while occupied, or the
//!     recycle-stack **next-free link** while free. A free slot has no live
//!     gid, so the two uses never coincide in time.
//!   - `occupancy: Vec<AtomicU64>` — one bit per slot, the authoritative
//!     free/occupied discriminator for the overlapped `counts` word.
//!   - `recycle_head: AtomicU64` + `hwm: AtomicU32` — the O(1) free list: a
//!     lock-free intrusive Treiber stack of freed slots (`recycle_head`) plus a
//!     high-water mark (`hwm`) that hands out never-used slots. No scan.
//!
//! Every [`ChunkGid`] carries an `Arc<ArenaRefcounts>` (shared with every
//! other gid in the same arena) plus an `i64` id. No per-gid heap
//! allocation. Clone/drop are single atomic ops on `counts[chunk_idx]`.
//! At 10M-token arena scale, this replaces 10M `Arc<GidInner>` heap
//! allocations with one ~600 KB contiguous `Vec<AtomicU16>` per arena.
//!
//! ### Allocation path
//!
//! `ArenaPool::allocate_n` iterates arenas in `arena_idx` order (skipping full
//! ones via the pool capacity bitmap). For each arena it claims slots in O(1):
//! pop the `recycle_head` stack if non-empty, else bump `hwm`. No per-slot
//! scan, so allocation cost is independent of how fragmented the arena is.
//!
//! ### Drop path
//!
//! `ChunkGid::drop` does `counts[chunk_idx].fetch_sub(1)`. On the `1→0`
//! last-drop the slot is freed: its `occupancy` bit is cleared and it is pushed
//! onto the `recycle_head` stack (its `counts` word becomes the next-free link).
//! Drops are ungated and only ever push; allocation is gated to a single popper,
//! which keeps the Treiber pop ABA-free.
//!
//! No mutex anywhere on the hot path. The pool's `RwLock<HashMap>` is
//! taken only on `register_arena` / `release_arena` (rare).
//!
//! ## What was replaced
//!
//! The previous design held the free set as a per-format `Mutex<BinaryHeap>`
//! plus `BTreeSet<arena_idx>` plus `HashMap<arena_idx, free_count>`.
//! Every alloc and drop took the per-format mutex; every gid was a
//! separate `Arc<GidInner>` heap allocation. The lock-free refcount-table
//! design collapses all of that into one contiguous `Vec<AtomicU16>` per
//! arena and lets drops touch a single cache line.

use ahash::{AHashMap, AHashSet};
use std::collections::BTreeMap;
use std::{
    collections::VecDeque,
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, RwLock,
    },
};
use strum::IntoEnumIterator;

use super::arena::ArenaKey;
use super::size_class::SizeClass;
use crate::kv_cache::chunked::types::{GID_STRIDE, TARGET_ARENA_BYTES};
use crate::kv_cache::ArenaLocation;

/// The class the pool's own convenience allocator and its unit tests use.
///
/// Rung 5 is **640 B** (`Q4_KS`), which puts 26,214 slots in a 16 MiB region —
/// well above the 15,420 of `Q8_0`'s 1088 B rung. The point is to exercise the
/// free list and the `u16` recycle links near their busiest, not to mirror the
/// most common sealed format, so a *small* class is the stronger choice here.
///
/// (This said "2048 B … holds `Q8_0`". Both halves were wrong: rung 5 is 640 B,
/// and 2048 B is the `F16`/`BF16` rung — `Q8_0` sits at 1088.)
const TEST_CLASS: SizeClass = SizeClass::at(5);

/// Per-arena refcount table. Lives behind an `Arc` shared by every
/// `ChunkGid` allocated from this arena. Lock-free: all mutation is via
/// atomics on `counts[chunk_idx]`, `occupancy`, and `recycle_head`/`hwm`.
#[derive(Debug)]
pub struct ArenaRefcounts {
    /// Dual-purpose per-slot word, disambiguated by the `occupancy` bit:
    ///   * **occupied** (`occupancy` bit set) → the slot's **refcount**
    ///     (`≥ 1`); `compare_exchange`/`fetch_add`/`fetch_sub` against it are
    ///     the real COW-share bookkeeping.
    ///   * **free** (`occupancy` bit clear) → the intrusive recycle-stack
    ///     **link**: the index of the next free slot, or `arena_chunks` for the
    ///     bottom of the stack. A free slot has no live `ChunkGid`, so nothing
    ///     ever reads it as a refcount — the two uses never overlap in time.
    ///
    /// `u16` holds either (arena chunk counts and indices are both far below
    /// 65536), so the free list needs no separate links array.
    counts: Vec<AtomicU16>,
    /// Occupancy: one BIT per slot, `set ⟺ slot is allocated`. With `counts`
    /// overlapped (refcount xor link) this is now the **authoritative** free/
    /// occupied discriminator (not just a scan hint): set on claim, cleared on
    /// the `1→0` last drop, `fetch_or`/`fetch_and` so different bits of a word
    /// compose. Read by `live_gids` to enumerate live slots.
    occupancy: Vec<AtomicU64>,
    /// Lock-free intrusive recycle stack of freed slots (Treiber). Links live
    /// in `counts` (a free slot's word = next-free index, or `arena_chunks` =
    /// bottom). Packed head: low 32 bits = top slot index (`arena_chunks` ⇒
    /// empty), high 32 bits = ABA version tag. `dec` pushes a freed slot; alloc
    /// pops — both O(1), no scan, fragmentation-immune. Allocs are gated
    /// (single popper), so ABA can't actually arise; the tag is belt-and-braces.
    recycle_head: AtomicU64,
    /// High-water mark: index of the next never-allocated slot. Fresh capacity
    /// comes from bumping this (so a new arena needs no O(A) free-list init —
    /// the "one range `[0, A)`"); only *freed* slots ride `recycle_head`.
    /// Written only by the gated allocator, so plain relaxed load/store.
    hwm: AtomicU32,
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
    /// Shared pool arena-capacity bitmap. `dec` sets this arena's bit on a
    /// full → non-full transition so the pool's `allocate_any` can find it via
    /// find-first-set instead of scanning every arena.
    capacity: Arc<CapacityBitmap>,
    /// Creation window guard: `true` from registration until the arena hands
    /// out its FIRST gid. `register_arena` releases the metadata lock before
    /// its caller allocates chunks or writes data, so a freshly-registered
    /// arena sits at `live == 0`, unprotected — and `try_tombstone` on another
    /// thread could free it (and recycle its INDEX to a different owner) while
    /// the creator is mid-allocation or mid-write. That is the "arena with
    /// active KV freed under an in-flight kernel" class: an illegal address
    /// when the memory unmaps, or silent cross-context KV contamination when
    /// the index/memory is re-tenanted. This flag closes the window at the
    /// ownership level: an arena is tombstoneable only after its creator has
    /// taken at least one slot (from then on `live > 0` protects it, and a
    /// later genuine drop to `live == 0` is legitimately reclaimable).
    /// Cleared with `Release` after the live increment; checked with `Acquire`
    /// before the live read, so observing `false` implies seeing `live ≥ 1`.
    creation_pending: AtomicBool,
}

impl ArenaRefcounts {
    fn new(
        arena_chunks: usize,
        arena_idx: usize,
        key: ArenaKey,
        pool_total_live: Arc<AtomicUsize>,
        capacity: Arc<CapacityBitmap>,
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
            // Empty stack: low 32 bits = `arena_chunks` sentinel, version 0.
            recycle_head: AtomicU64::new(arena_chunks as u64),
            hwm: AtomicU32::new(0),
            live: AtomicUsize::new(0),
            pool_total_live,
            arena_chunks,
            arena_idx,
            key,
            capacity,
            creation_pending: AtomicBool::new(true),
        }
    }

    /// Whether this arena is still inside its creation window (registered but
    /// no gid allocated yet). `Acquire` pairs with the `Release` clear in
    /// [`Self::occupy`]: a `false` here guarantees the first slot's live
    /// increment is visible.
    #[inline]
    fn creation_pending(&self) -> bool {
        self.creation_pending.load(Ordering::Acquire)
    }

    /// End the creation window without claiming a slot.
    ///
    /// The window keeps a tombstoner off an arena index whose creator is still
    /// working on it — recycling the index under the creator leaves storage and
    /// pool disagreeing about its format, which is cross-context KV
    /// contamination rather than a clean fault. What closes the window is the
    /// creator *finishing*, and for every allocate-on-demand path that is its
    /// first [`Self::occupy`], so the two coincide and the flag can ride along
    /// with the first claim.
    ///
    /// An arena created **ahead of** demand has no first claim to ride:
    /// `create_deferred_arenas` stamps a slab for a class that asked for one
    /// during a wave, and nothing occupies it until the demand arrives. Left to
    /// `occupy`, its window never closes — and an arena that is empty is
    /// counted by [`ArenaPool::has_reclaimable`] and refused by
    /// [`ArenaPool::try_tombstone`] forever, which is exactly the "pool reports
    /// memory it cannot hand over" wedge. So that creator closes the window
    /// itself, once the slab is in storage and it is done with the index.
    #[inline]
    fn end_creation_window(&self) {
        self.creation_pending.store(false, Ordering::Release);
    }

    /// Unpack the recycle head into `(top_slot_idx, version)`; `top == arena_chunks`
    /// means the stack is empty.
    #[inline]
    fn head_parts(head: u64) -> (usize, u64) {
        ((head & 0xFFFF_FFFF) as usize, head >> 32)
    }

    /// Pack a `(top_slot_idx, version)` into a recycle-head word.
    #[inline]
    fn head_pack(idx: usize, version: u64) -> u64 {
        ((version & 0xFFFF_FFFF) << 32) | (idx as u64 & 0xFFFF_FFFF)
    }

    /// Whether every slot is claimed (`live == arena_chunks`). The authoritative
    /// full check the pool capacity bit mirrors.
    #[inline]
    fn is_full(&self) -> bool {
        self.live.load(Ordering::Acquire) >= self.arena_chunks
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

    /// Mark slot `i` free in the occupancy discriminator (clear its bit).
    ///
    /// Atomic `fetch_and` so it composes with a concurrent `set_occupied`.
    /// **Release**: `dec` calls this before overwriting `counts[i]` with the
    /// recycle-stack link, so a reader that Acquire-observes the cleared bit
    /// (slot free) never still sees the stale refcount in the word.
    #[inline]
    fn set_free(&self, i: usize) {
        self.occupancy[i / 64].fetch_and(!(1u64 << (i % 64)), Ordering::Release);
    }

    /// Finish claiming slot `i`: overwrite its `counts` word (was a free-list
    /// link) with refcount 1, mark it occupied, and bump the live counters.
    /// Runs under `alloc_gate`, and no `ChunkGid` for `i` exists yet, so no one
    /// races these writes.
    #[inline]
    fn occupy(&self, i: usize) {
        self.counts[i].store(1, Ordering::Relaxed);
        self.set_occupied(i);
        self.live.fetch_add(1, Ordering::Relaxed);
        self.pool_total_live.fetch_add(1, Ordering::Relaxed);
        // End the creation window AFTER the live increment (Release), so a
        // tombstoner that Acquire-observes `creation_pending == false` also
        // sees `live ≥ 1` — there is no interleaving where the arena looks
        // both "past creation" and "empty" while its first chunk is in flight.
        self.end_creation_window();
    }

    /// Claim a free slot in O(1): pop the recycle stack, else bump the high-
    /// water mark. No scan — fully fragmentation-immune. Callers serialize this
    /// via `ArenaPool::alloc_gate`, so there is exactly one popper; concurrent
    /// `dec` drops only ever *push*, which is what makes the single-popper
    /// Treiber pop ABA-free (a slot can't be popped and re-pushed under us).
    fn try_claim_one(&self) -> Option<usize> {
        // 1) Reuse a freed slot from the recycle stack.
        loop {
            let head = self.recycle_head.load(Ordering::Acquire);
            let (top, version) = Self::head_parts(head);
            if top == self.arena_chunks {
                break; // empty — fall through to the high-water mark.
            }
            // The next-link lives in the free slot's `counts` word. Reading it as
            // a link is sound: `top` is on the stack, so the pushing `dec`'s
            // `Release` link store happens-before this `Acquire` head load.
            let next = self.counts[top].load(Ordering::Acquire) as usize;
            let new_head = Self::head_pack(next, version.wrapping_add(1));
            if self
                .recycle_head
                .compare_exchange_weak(head, new_head, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.occupy(top);
                return Some(top);
            }
            // Lost the CAS to a concurrent push — retry with the fresh head.
        }
        // 2) No recycled slot: hand out the next never-used slot. `hwm` is
        //    written only by the gated allocator, so a plain load/store is enough.
        let h = self.hwm.load(Ordering::Relaxed) as usize;
        if h < self.arena_chunks {
            self.hwm.store((h + 1) as u32, Ordering::Relaxed);
            self.occupy(h);
            return Some(h);
        }
        None // full: recycle stack empty and high-water mark exhausted.
    }

    /// Claim `len` CONSECUTIVE slots, returning the first index. High-water
    /// mark only — recycled singleton slots are never consecutive-by-contract.
    /// Contiguous runs give the QREL / fused-select walk better spatial
    /// locality; they are not required for correctness (each band is addressed
    /// through its own gid — `resolve_band_source`). Callers serialize via
    /// `alloc_gate`.
    fn try_claim_run(&self, len: usize) -> Option<usize> {
        let h = self.hwm.load(Ordering::Relaxed) as usize;
        if h + len <= self.arena_chunks {
            self.hwm.store((h + len) as u32, Ordering::Relaxed);
            for i in 0..len {
                self.occupy(h + i);
            }
            return Some(h);
        }
        None
    }

    /// Whether the never-used tail can still fit a run of `len` slots.
    #[inline]
    fn run_fits(&self, len: usize) -> bool {
        (self.hwm.load(Ordering::Relaxed) as usize) + len <= self.arena_chunks
    }

    /// Increment a slot's refcount — called by `ChunkGid::clone`. The
    /// slot must already be allocated (count ≥ 1); we never need
    /// Acquire semantics because the data is already visible.
    #[inline]
    fn inc(&self, chunk_idx: usize) {
        self.counts[chunk_idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement a slot's refcount — called by `ChunkGid::drop`. On the `1→0`
    /// last drop, free the slot: clear its occupancy bit, push it onto the
    /// recycle stack (its `counts` word becomes the next-free link), and
    /// decrement the live counters (per-arena and pool-wide).
    #[inline]
    fn dec(&self, chunk_idx: usize) {
        let prev = self.counts[chunk_idx].fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            let prev_live = self.live.fetch_sub(1, Ordering::Relaxed);
            self.pool_total_live.fetch_sub(1, Ordering::Relaxed);
            // Mark the slot free in the occupancy discriminator FIRST, so a
            // reader never sees it "occupied" while `counts` already holds a
            // link, nor "free" while `counts` still holds the stale refcount.
            self.set_free(chunk_idx);
            // Push the freed slot onto the recycle stack. Its next-link lives in
            // its own (now-free) `counts` word; the `Release` head CAS publishes
            // that link store to whichever claimer later pops it. Only pushes are
            // concurrent here (drops are ungated); the single gated popper means
            // the popped slot's link can't change under a claimer, so no ABA.
            loop {
                let head = self.recycle_head.load(Ordering::Acquire);
                let (top, version) = Self::head_parts(head);
                self.counts[chunk_idx].store(top as u16, Ordering::Release);
                let new_head = Self::head_pack(chunk_idx, version.wrapping_add(1));
                if self
                    .recycle_head
                    .compare_exchange_weak(head, new_head, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
            // Publish capacity to the pool bitmap AFTER the push (the slot is now
            // poppable). Only the drop that takes the arena full → non-full sets
            // it (`prev_live` was the full count), at most once per fill/empty
            // cycle. A wrongly-cleared bit (this racing an alloc's fill-clear)
            // can only *hide* capacity, which `allocate_any`'s fallback resync
            // recovers — it can never leak a slot permanently.
            if prev_live == self.arena_chunks {
                self.capacity.set(self.arena_idx);
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
                let chunk_idx = (self.id as usize) % GID_STRIDE;
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
        self.id as usize / GID_STRIDE
    }

    /// Chunk offset within its arena.
    #[inline]
    pub fn chunk_idx(&self) -> usize {
        self.id as usize % GID_STRIDE
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
                let chunk_idx = (self.id as usize) % GID_STRIDE;
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
                let chunk_idx = (self.id as usize) % GID_STRIDE;
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
/// Per-format bitmap: bit `i` set ⇒ arena `i` of this format has ≥1 free slot.
/// The pool's O(1) "which arena has capacity" index — it replaces the per-alloc
/// walk over every arena that was the pressure-regime `alloc` bottleneck
/// (`allocate_any` used to iterate the whole `tables` map, O(num_arenas), for
/// every one of the ~1M allocations a drain pass issues). Fully lock-free: `dec`
/// sets a bit on a full→non-full transition, alloc clears it on non-full→full
/// and reads it via find-first-set. Fixed-size so `dec` never contends a resize;
/// sized from a 512 GiB VRAM ceiling / arena size, far past any single card, so
/// `arena_idx` (VRAM-bounded — indices are recycled on tombstone) never exceeds
/// it. `find-first-set` returns the lowest such arena, preserving the lowest-
/// first packing compaction relies on.
#[derive(Debug)]
struct CapacityBitmap {
    words: Box<[AtomicU64]>,
}

impl CapacityBitmap {
    fn new() -> Self {
        let max_arenas = ((512usize << 30) / TARGET_ARENA_BYTES).max(4096);
        let words = (0..max_arenas.div_ceil(64))
            .map(|_| AtomicU64::new(0))
            .collect();
        Self { words }
    }

    #[inline]
    fn set(&self, arena_idx: usize) {
        if let Some(w) = self.words.get(arena_idx >> 6) {
            w.fetch_or(1u64 << (arena_idx & 63), Ordering::Release);
        }
    }

    #[inline]
    fn clear(&self, arena_idx: usize) {
        if let Some(w) = self.words.get(arena_idx >> 6) {
            w.fetch_and(!(1u64 << (arena_idx & 63)), Ordering::Release);
        }
    }

    /// Lowest arena index whose has-capacity bit is set, or `None` if all clear.
    #[inline]
    fn first_set(&self) -> Option<usize> {
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
    /// Bit `i` set ⇒ arena `i` has ≥1 free slot. `allocate_any` finds the
    /// lowest such arena via find-first-set instead of walking every arena.
    capacity: Arc<CapacityBitmap>,
}

impl ArenaPool {
    fn new(class: SizeClass) -> Self {
        Self {
            tables: RwLock::new(BTreeMap::new()),
            total_arenas: AtomicUsize::new(0),
            total_live: Arc::new(AtomicUsize::new(0)),
            arena_chunks: class.chunks_per_region(),
            alloc_gate: Mutex::new(()),
            capacity: Arc::new(CapacityBitmap::new()),
        }
    }

    /// Register a new arena with the pool — creates its refcount table.
    fn register_arena(&self, arena_idx: usize, key: ArenaKey) -> Arc<ArenaRefcounts> {
        let table = Arc::new(ArenaRefcounts::new(
            self.arena_chunks,
            arena_idx,
            key,
            Arc::clone(&self.total_live),
            Arc::clone(&self.capacity),
        ));
        {
            let mut tables = self.tables.write().unwrap();
            tables.insert(arena_idx, Arc::clone(&table));
        }
        self.total_arenas.fetch_add(1, Ordering::Relaxed);
        // A fresh arena is all free — mark it available. Ordered after the
        // `tables` insert so a claimer that sees the bit also finds the table.
        self.capacity.set(arena_idx);
        table
    }

    /// Allocate a single gid from any arena. Iterates arenas in
    /// `arena_idx` order (lowest first) so live data clusters in low
    /// indices, keeping compaction effective.
    /// Claim `len` CONSECUTIVE slots in one arena of this pool, returning the
    /// first raw gid and the arena's refcount table. Only the never-used
    /// high-water tail of an arena can host a run (see
    /// [`ArenaRefcounts::try_claim_run`]); arenas whose tail is exhausted are
    /// skipped, and `None` means the caller must register a fresh arena.
    fn allocate_run(&self, len: usize) -> Option<(i64, Arc<ArenaRefcounts>)> {
        let _gate = self.alloc_gate.lock().unwrap();
        let stride = GID_STRIDE;
        let tables = self.tables.read().unwrap();
        let mut indices: Vec<usize> = tables.keys().copied().collect();
        indices.sort_unstable();
        for arena_idx in indices {
            let table = &tables[&arena_idx];
            if !table.run_fits(len) {
                continue;
            }
            if let Some(first) = table.try_claim_run(len) {
                if table.is_full() {
                    self.capacity.clear(arena_idx);
                }
                return Some(((arena_idx * stride + first) as i64, Arc::clone(table)));
            }
        }
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
    fn allocate_run_in(&self, arena_idx: usize, len: usize) -> Option<(i64, Arc<ArenaRefcounts>)> {
        // Same gate as every claim walk: `try_claim_run` is load-then-store on
        // `hwm` and is only sound serialized.
        let _gate = self.alloc_gate.lock().unwrap();
        let stride = GID_STRIDE;
        let tables = self.tables.read().unwrap();
        let table = tables.get(&arena_idx)?;
        let first = table.try_claim_run(len)?;
        if table.is_full() {
            self.capacity.clear(arena_idx);
        }
        Some(((arena_idx * stride + first) as i64, Arc::clone(table)))
    }

    fn allocate_any(&self) -> Option<(i64, Arc<ArenaRefcounts>)> {
        // Serialize the claiming walk: only one thread scans this pool's
        // `counts` arrays at a time, so `try_claim_one` can probe occupancy
        // with a relaxed load (no per-slot locked RMW) without 128 threads
        // ping-ponging the same cache lines. Drops stay lock-free.
        let _gate = self.alloc_gate.lock().unwrap();
        let stride = GID_STRIDE;
        let tables = self.tables.read().unwrap();
        // Fast path: the capacity bitmap points at the lowest arena with a free
        // slot (find-first-set = lowest-first packing), skipping the full-arena
        // prefix that made this an O(num_arenas) walk per alloc at pressure. A
        // set bit can be stale (the arena filled since it was set) → the claim
        // fails → clear it and try the next set bit.
        while let Some(arena_idx) = self.capacity.first_set() {
            match tables.get(&arena_idx) {
                Some(table) => {
                    if let Some(chunk_idx) = table.try_claim_one() {
                        if table.is_full() {
                            self.capacity.clear(arena_idx);
                        }
                        return Some(((arena_idx * stride + chunk_idx) as i64, Arc::clone(table)));
                    }
                    // Stale set bit — arena is full. Clear and try the next.
                    self.capacity.clear(arena_idx);
                }
                // Bit set for a tombstoned arena (register/release race) — clear.
                None => self.capacity.clear(arena_idx),
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
        for (&arena_idx, table) in tables.iter() {
            if table.free_count() > 0 {
                self.capacity.set(arena_idx);
                recovered = true;
            }
        }
        if recovered {
            if let Some(arena_idx) = self.capacity.first_set() {
                if let Some(table) = tables.get(&arena_idx) {
                    if let Some(chunk_idx) = table.try_claim_one() {
                        if table.is_full() {
                            self.capacity.clear(arena_idx);
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
    fn allocate_n(&self, n: usize) -> Vec<(i64, Arc<ArenaRefcounts>)> {
        if n == 0 {
            return Vec::new();
        }
        let mut out: Vec<(i64, Arc<ArenaRefcounts>)> = Vec::with_capacity(n);

        // One gate acquisition for the whole bulk claim (see `allocate_any`).
        let _gate = self.alloc_gate.lock().unwrap();
        // Same lowest-first BTreeMap walk as `allocate_any`, draining each
        // arena before moving up — one read lock, no index collect + sort.
        let stride = GID_STRIDE;
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
    fn try_tombstone(&self, protected_arenas: &AHashSet<usize>) -> Option<usize> {
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
        // Lowest-index fully-free, non-protected arena. `creation_pending` is
        // checked FIRST: its Acquire load pairs with `occupy`'s Release clear,
        // so an arena observed past its creation window is guaranteed to show
        // its first allocation in `live_count` — a freshly-registered arena
        // whose creator hasn't allocated yet can never be tombstoned (freeing
        // it would unmap memory an in-flight kernel writes, and recycle its
        // index to a second owner: cross-context KV contamination).
        let candidate = tables
            .iter()
            .filter(|(idx, t)| {
                !t.creation_pending() && t.live_count() == 0 && !protected_arenas.contains(idx)
            })
            .map(|(&idx, _)| idx)
            .min()?;
        drop(tables);

        let mut tables = self.tables.write().unwrap();
        tables.remove(&candidate);
        self.total_arenas.fetch_sub(1, Ordering::Relaxed);
        // Arena gone — clear its capacity bit so `allocate_any` doesn't chase a
        // dangling index (it self-heals via the `tables.get` miss anyway).
        self.capacity.clear(candidate);
        Some(candidate)
    }

    /// Force-remove an arena's table regardless of whether it's empty.
    /// Used by the legacy `release_arena` path after a manual gid drain.
    fn force_release(&self, arena_idx: usize) {
        let mut tables = self.tables.write().unwrap();
        if tables.remove(&arena_idx).is_some() {
            self.total_arenas.fetch_sub(1, Ordering::Relaxed);
            self.capacity.clear(arena_idx);
        }
    }

    fn has_reclaimable(&self) -> bool {
        let tables = self.tables.read().unwrap();
        tables.values().any(|t| t.live_count() == 0)
    }

    /// Close `arena_idx`'s creation window. No-op if the arena is already gone
    /// or already past creation — see [`ArenaRefcounts::end_creation_window`].
    fn finish_creation(&self, arena_idx: usize) {
        let tables = self.tables.read().unwrap();
        if let Some(t) = tables.get(&arena_idx) {
            t.end_creation_window();
        }
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
    // Locations × size classes — 14 entries, down from ~58 under per-format
    // pools. That collapse is the whole point: every format sharing a class
    // now shares one pool and one free list, so a slot freed by any of them is
    // allocatable by all of them (`docs/archived/arena_unification.md` §3.4).
    let mut pools = AHashMap::with_capacity(ArenaLocation::iter().count() * SizeClass::COUNT);
    for location in ArenaLocation::iter() {
        for class in SizeClass::all() {
            pools.insert(ArenaKey::new(class, location), ArenaPool::new(class));
        }
    }
    pools
}

/// Occupancy of one size class's GPU arenas.
///
/// `reserved` counts whole ~16 MiB slabs ([`TARGET_ARENA_BYTES`]); `live`
/// counts occupied chunk slots at this class's stride. Both are GidPool
/// arena-slab quantities, distinct from the CUDA stream-ordered pool's
/// `reserved`/`used` (which include segment slack the GidPool never sees) —
/// report them on their own line, not as a partition of the CUDA-pool gap.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClassOccupancy {
    /// Slot stride for this class, in bytes — its identity in a report.
    pub slot_bytes: usize,
    pub arenas: usize,
    pub reserved_bytes: usize,
    pub live_bytes: usize,
}

/// GPU arena occupancy per size class — the diagnostic the compress-to-free
/// relief rung is judged by.
///
/// This replaces the old float-vs-quant split, which cannot be computed any
/// more and was never quite the right question: an arena has no format, and
/// what the rung actually moves is occupancy *down the ladder* as bands
/// compress into smaller classes. A working rung shows the large classes'
/// live bytes falling while the small classes' rise by less.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuArenaClassStats {
    /// One entry per rung of the ladder, in ascending stride order.
    pub classes: [ClassOccupancy; SizeClass::COUNT],
}

impl GpuArenaClassStats {
    /// Total GPU arenas across every class.
    pub fn total_arenas(&self) -> usize {
        self.classes.iter().map(|c| c.arenas).sum()
    }

    /// Total slab bytes reserved across every class.
    pub fn total_reserved_bytes(&self) -> usize {
        self.classes.iter().map(|c| c.reserved_bytes).sum()
    }

    /// Total bytes in occupied slots across every class.
    pub fn total_live_bytes(&self) -> usize {
        self.classes.iter().map(|c| c.live_bytes).sum()
    }
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
        tables.values().any(|t| t.run_fits(len))
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

    /// Find a fully-free arena of this format and release it.
    pub fn next_tombstone(&self, key: ArenaKey) -> Option<usize> {
        let mut state = self.inner.metadata.lock().unwrap();
        let pool = self.inner.pools.get(&key)?;
        let arena_idx = pool.try_tombstone(&state.protected_arenas)?;
        state.arena_registry[arena_idx] = None;
        state.free_arenas.push_back(arena_idx);
        Some(arena_idx)
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
            let key = state.arena_registry.get(arena_idx).and_then(|k| *k);
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

#[cfg(test)]
mod tests {
    use super::*;
    fn float_key() -> ArenaKey {
        ArenaKey::new(TEST_CLASS, ArenaLocation::Gpu)
    }

    fn test_arena_chunks() -> usize {
        TEST_CLASS.chunks_per_region()
    }

    /// **An arena created ahead of demand is reclaimable once its creator is
    /// done with it.**
    ///
    /// The creation window keeps a tombstoner off an index whose creator is
    /// still working on it, and every allocate-on-demand path closes that
    /// window by claiming its first chunk — so for those, "past creation" and
    /// "has been used" are the same instant and the flag can ride along with
    /// the claim. `create_deferred_arenas` stamps a slab *before* the demand
    /// arrives and has no first claim to ride, so left to `occupy` its window
    /// never closes. An arena stuck inside its window is counted by
    /// `has_reclaimable` and refused by `try_tombstone` for the life of the
    /// process, which is the pool reporting memory it cannot hand over.
    ///
    /// Measured through the substrate before this closed: one 16 MiB region
    /// stranded per pool per persistence pass, every slot in it free.
    #[test]
    fn an_arena_made_ahead_of_demand_is_reclaimable_once_its_creator_finishes() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let idx = pool.register_arena(key);

        // Registered and never claimed from: the pool counts it as recoverable…
        assert!(
            pool.has_reclaimable(),
            "an arena with every slot free is reported reclaimable"
        );
        // …and refuses to hand it over, because its creation window is open.
        // That refusal is correct while a creator still holds the index — a
        // recycled index under a live creator is cross-context contamination,
        // not a clean fault.
        assert_eq!(
            pool.next_tombstone(key),
            None,
            "an arena inside its creation window must not be tombstoned"
        );

        // The creator finishes without ever claiming a chunk — exactly what
        // pre-creating a slab for a class that asked for one looks like.
        pool.finish_creation(key, idx);

        assert_eq!(
            pool.next_tombstone(key),
            Some(idx),
            "an empty arena whose creator has finished must be reclaimable, or \
             it pins its region until the process exits"
        );
    }

    /// The explicit close did not replace the implicit one: a creator that
    /// claims a chunk still closes its own window, which is what every
    /// allocate-on-demand path relies on.
    #[test]
    fn claiming_a_chunk_still_closes_the_creation_window() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let idx = pool.register_arena(key);

        let gid = pool.allocate_for(key).expect("fresh arena serves a claim");
        assert_eq!(gid.arena_idx(), idx);
        assert_eq!(
            pool.next_tombstone(key),
            None,
            "an occupied arena is not reclaimable"
        );

        drop(gid);
        assert_eq!(
            pool.next_tombstone(key),
            Some(idx),
            "a claimed-then-dropped arena is reclaimable with no explicit \
             finish_creation — `occupy` closes the window for its creator"
        );
    }

    /// The fix for the "fresh arena cannot fit palette run" race: a run claimed
    /// BY INDEX from a just-registered arena must succeed even when the global
    /// walk would have been raced, and must fail cleanly once that arena's tail
    /// is consumed (the caller then registers another).
    #[test]
    fn targeted_run_claim_hits_the_registered_arena() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();
        let idx = pool.register_arena(key);

        // Simulate the race: a rival's global walk consumes most of the fresh
        // arena's tail before our targeted claim.
        let rival = pool
            .allocate_run_for(key, cap - 2)
            .expect("rival run fits the fresh arena");
        assert_eq!(rival.len(), cap - 2);

        // The global walk can no longer fit 3 — but the targeted claim reports
        // that the SPECIFIC arena is exhausted (None), not a phantom "no arena
        // anywhere", so the caller knows to register another…
        assert!(pool.allocate_run_for(key, 3).is_none());
        assert!(pool.allocate_run_for_in(key, idx, 3).is_none());

        // …and a run that still fits the tail lands in exactly that arena.
        let run = pool
            .allocate_run_for_in(key, idx, 2)
            .expect("2 slots remain at the high-water tail");
        assert_eq!(run.len(), 2);
        assert!(run.iter().all(|g| g.arena_idx() == idx));

        // A fresh registration + targeted claim succeeds for the full length —
        // the loop the allocator runs.
        let idx2 = pool.register_arena(key);
        let run2 = pool
            .allocate_run_for_in(key, idx2, cap)
            .expect("fresh arena serves a full-capacity run");
        assert_eq!(run2.len(), cap);
        assert!(run2.iter().all(|g| g.arena_idx() == idx2));

        // Unknown arena index: None, never a panic.
        assert!(pool.allocate_run_for_in(key, 9999, 1).is_none());
    }

    /// An empty arena is always reclaimed, however full the rest of the pool is.
    ///
    /// A 10 % free-headroom guard used to hold it back to stop create/destroy
    /// churn, and needed a `force` bypass because its condition ("the pool is
    /// nearly full") was exactly the state reclaim was being asked to fix.
    /// Under the reservation an arena's storage is a region handle, so there is
    /// no churn left to guard: this is the state that used to refuse.
    #[test]
    fn an_empty_arena_is_reclaimed_even_with_no_headroom_left() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();

        // Two arenas; fill BOTH completely, then free exactly one arena's worth.
        // Live is then `cap` across 2 arenas, so releasing one leaves zero free
        // slots — the case the old guard refused.
        pool.register_arena(key);
        let a: Vec<_> = (0..cap).map(|_| pool.allocate_for(key).unwrap()).collect();
        pool.register_arena(key);
        let b: Vec<_> = (0..cap).map(|_| pool.allocate_for(key).unwrap()).collect();
        assert_eq!(a.len() + b.len(), cap * 2);

        let freed_idx = b[0].arena_idx();
        assert!(
            b.iter().all(|g| g.arena_idx() == freed_idx),
            "b filled one arena"
        );
        drop(b);

        assert_eq!(
            pool.next_tombstone(key),
            Some(freed_idx),
            "the empty arena is reclaimed"
        );

        drop(a);
    }

    #[test]
    fn test_register_and_allocate() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let arena_idx = pool.register_arena(key);
        assert_eq!(arena_idx, 0);

        let gid1 = pool.allocate_for(key).unwrap();
        assert_eq!(gid1.raw(), 0);
        let gid2 = pool.allocate_for(key).unwrap();
        assert_eq!(gid2.raw(), 1);
    }

    /// Occupancy is reported **per size class**, and a class's live bytes are
    /// its slot count times its stride — whatever formats happen to occupy it.
    #[test]
    fn gpu_class_stats_report_per_class_occupancy() {
        let pool = ChunkGidPool::new();
        let small = ArenaKey::new(SizeClass::at(0), ArenaLocation::Gpu);
        let large = ArenaKey::new(SizeClass::at(6), ArenaLocation::Gpu);

        // Two arenas in the small class with three live slots; one arena in
        // the large class with five.
        pool.register_arena(small);
        pool.register_arena(small);
        pool.register_arena(large);
        let _s: Vec<_> = (0..3).map(|_| pool.allocate_for(small).unwrap()).collect();
        let _l: Vec<_> = (0..5).map(|_| pool.allocate_for(large).unwrap()).collect();

        let stats = pool.gpu_class_stats();
        let row = |c: SizeClass| stats.classes[c.index()];

        assert_eq!(row(small.class).slot_bytes, small.slot_stride());
        assert_eq!(row(small.class).arenas, 2);
        assert_eq!(row(small.class).reserved_bytes, 2 * TARGET_ARENA_BYTES);
        assert_eq!(row(small.class).live_bytes, 3 * small.slot_stride());

        assert_eq!(row(large.class).arenas, 1);
        assert_eq!(row(large.class).reserved_bytes, TARGET_ARENA_BYTES);
        assert_eq!(row(large.class).live_bytes, 5 * large.slot_stride());

        assert_eq!(stats.total_arenas(), 3);
        assert_eq!(
            stats.total_live_bytes(),
            3 * small.slot_stride() + 5 * large.slot_stride()
        );
    }

    /// **Two formats sharing a class share a pool.** This is the property the
    /// whole initiative exists to obtain, and it is visible here: allocating
    /// for two different formats that map to one class grows a single row, not
    /// two, and never registers a second arena.
    #[test]
    fn formats_sharing_a_class_share_one_pool() {
        use crate::kv_cache::{KvFormat, QuantFormat};
        const ELEMS: usize = 1024;
        let pool = ChunkGidPool::new();
        // Q4_1 and Q4_KS are both 640 B payloads, so one rung, one key.
        let a = ArenaKey::for_format(
            KvFormat::Quantized(QuantFormat::Q4_1),
            ELEMS,
            ArenaLocation::Gpu,
        )
        .unwrap();
        let b = ArenaKey::for_format(
            KvFormat::Quantized(QuantFormat::Q4_KS),
            ELEMS,
            ArenaLocation::Gpu,
        )
        .unwrap();
        assert_eq!(a, b, "the two formats must resolve to one key");

        pool.register_arena(a);
        let g1 = pool.allocate_for(a).expect("first format claims a slot");
        let g2 = pool
            .allocate_for(b)
            .expect("second format claims from the SAME pool");
        assert_eq!(g1.arena_idx(), g2.arena_idx());

        let stats = pool.gpu_class_stats();
        assert_eq!(stats.total_arenas(), 1, "one arena serves both formats");
        assert_eq!(
            stats.classes[a.class.index()].live_bytes,
            2 * a.slot_stride()
        );
    }

    #[test]
    fn gpu_class_stats_on_an_empty_pool_are_zero() {
        // A pool with registered-but-unallocated classes reports nothing: the
        // per-wave diagnostic must not count preallocated empty pool-table
        // entries as resident arenas.
        let stats = ChunkGidPool::new().gpu_class_stats();
        assert_eq!(stats.total_arenas(), 0);
        assert_eq!(stats.total_reserved_bytes(), 0);
        assert_eq!(stats.total_live_bytes(), 0);
        // The slot_bytes column is still populated, so a report shows the
        // whole ladder rather than a ragged subset.
        for (i, c) in stats.classes.iter().enumerate() {
            assert_eq!(c.slot_bytes, SizeClass::at(i).bytes());
        }
    }

    #[test]
    fn test_gid_drop_returns_to_pool() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key);

        let gid1 = pool.allocate_for(key).unwrap();
        let gid1_raw = gid1.raw();
        drop(gid1);

        // After drop, the slot is free again — total free == capacity.
        assert_eq!(pool.free_list_len_for(key), test_arena_chunks());

        // Allocating again should reuse it — the freed slot is popped straight
        // back off the recycle stack, so the same gid comes out.
        let gid2 = pool.allocate_for(key).unwrap();
        assert_eq!(gid2.raw(), gid1_raw);
    }

    #[test]
    fn test_gid_clone_no_early_return() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key);

        let gid1 = pool.allocate_for(key).unwrap();
        let gid1_clone = gid1.clone();
        let gid1_raw = gid1.raw();
        let free_after_alloc = pool.free_list_len_for(key);

        // Drop original — clone still holds a refcount.
        drop(gid1);
        assert_eq!(pool.free_list_len_for(key), free_after_alloc);

        // Drop clone — now freed.
        drop(gid1_clone);
        assert_eq!(pool.free_list_len_for(key), free_after_alloc + 1);

        let gid2 = pool.allocate_for(key).unwrap();
        assert_eq!(gid2.raw(), gid1_raw);
    }

    #[test]
    fn test_strong_count_tracks_clones() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.register_arena(key);
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
        let a = pool.register_arena(key);
        let b = pool.register_arena(key);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        // Each arena contributes `arena_chunks` of capacity.
        assert_eq!(pool.free_list_len_for(key), test_arena_chunks() * 2);
    }

    #[test]
    fn can_reclaim_arena_only_when_a_whole_arena_is_recoverable() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        // One arena holding a live chunk: less than a whole arena of free space,
        // so a forced compaction can release nothing.
        pool.register_arena(key);
        let _a0 = pool.allocate_from_arena(key, 0).unwrap();
        assert!(
            !pool.can_reclaim_arena(),
            "1 arena with a live chunk: nothing whole to reclaim"
        );
        // Add a second, empty arena: a whole arena's worth of free space is now
        // recoverable (needed = ceil(1 live / arena_chunks) = 1, of 2 arenas).
        pool.register_arena(key);
        assert!(
            pool.can_reclaim_arena(),
            "2 arenas, 1 live chunk: one whole arena is reclaimable"
        );
    }

    /// Hammer the capacity-bitmap allocator with concurrent lock-free drops
    /// racing gated allocs, filling and emptying arenas so the full↔non-full
    /// bit transitions (and their races with alloc's fill-clear) actually fire.
    /// Two invariants are asserted:
    ///   * **no double-allocation** — two threads claiming the same slot would
    ///     drive its refcount below zero on the second drop, panicking in `dec`
    ///     (`refcount underflow`), so the test would fail with that panic;
    ///   * **no permanently hidden capacity** — after every batch is dropped the
    ///     pool is fully free, so draining it must yield *exactly* the total free
    ///     count; a wrongly-cleared bit the fallback failed to recover would make
    ///     the drain come up short.
    #[test]
    fn concurrent_alloc_drop_recovers_all_capacity() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(ChunkGidPool::new());
        let key = float_key();
        pool.register_arena(key);

        let n_threads = 12;
        let batch = 400; // batches of live gids so arenas fill and empty
        let rounds = 40;
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    for _ in 0..rounds {
                        let mut held = Vec::with_capacity(batch);
                        for _ in 0..batch {
                            // Mirror `alloc_chunk_for_key`: register on exhaustion.
                            let g = loop {
                                if let Some(g) = pool.allocate_for(key) {
                                    break g;
                                }
                                pool.register_arena(key);
                            };
                            held.push(g);
                        }
                        // Drop the whole batch — lock-free `dec`s racing other
                        // threads' gated allocs (and their bit set/clear).
                        drop(held);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        // Everything dropped ⇒ the pool is fully free. Drain it and prove every
        // free slot is reachable via the bitmap fast path + the fallback scan.
        let cap = pool.free_list_len_for(key);
        assert!(cap > 0);
        let mut drained = Vec::new();
        while let Some(g) = pool.allocate_for(key) {
            drained.push(g);
        }
        assert_eq!(
            drained.len(),
            cap,
            "bitmap + fallback must reach every free slot after concurrent churn"
        );
    }
}
