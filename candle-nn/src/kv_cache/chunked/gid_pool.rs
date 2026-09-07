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
//! `ArenaPool::allocate_n` iterates arenas in **span-address order** — lowest
//! region first, which is what packs live chunks toward the bottom of the
//! reservation so the arenas above drain and give their regions back (see
//! `ArenaRefcounts::rank`). For each arena it claims slots in O(1):
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
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::{
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
    /// Where this arena sits in **span-address order**: its bit position in
    /// [`Self::capacity`] and its key in [`ArenaPool::by_rank`].
    ///
    /// A GPU arena occupies exactly one region and `REGION_BYTES` is the region
    /// stride, so the region index *is* the address order — `region_base +
    /// idx * REGION_BYTES` is monotonic in `idx`. Ranked by it, find-first-set
    /// returns the arena lowest in the span. Before the slab is carved there is
    /// no region yet, so the arena ranks at `UNRANKED_BASE + arena_idx`: past
    /// every real region, hence last resort, and still unique so no two arenas
    /// ever collide on one rank. [`ArenaPool::set_rank`] moves it down the
    /// moment the region is known.
    ///
    /// # Why the bitmap is not indexed by `arena_idx`
    ///
    /// Find-first-set returns the *lowest bit*, so whatever the bit position
    /// means is what allocation biases toward. Indexed by `arena_idx` it packed
    /// chunks into the lowest-numbered arena — and an index says nothing about
    /// where in the span that arena's bytes are, because arena indices and
    /// regions come from separate free lists and the region pool is shared with
    /// the recurrent stores and the wave tier. Chunks concentrated into arenas
    /// sitting anywhere, so churn never drained the *top* of the span and
    /// `live_watermark` — which caps both weight growth and the transient tier —
    /// stayed where the high-water mark had left it.
    ///
    /// Ranked by region, the same find-first-set biases every claim toward the
    /// leftmost arena. High arenas stop receiving chunks, drain as their chunks
    /// die, and their regions go back from the top.
    ///
    /// Held here, beside the fields `dec` already touches, so the drop path
    /// stays lock-free: one relaxed load, no map lookup.
    ///
    /// A rank read can be an instant stale across [`ArenaPool::set_rank`] — a
    /// concurrent drop may set the bit at the arena's previous position. That
    /// is a *placement* inaccuracy, never a correctness one: the bit is only a
    /// hint about where free space is, and a vacated position resolves through
    /// the same miss-and-clear path `allocate_any` already runs for a
    /// tombstoned arena.
    rank: AtomicUsize,
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
            // No region carved yet — rank past every real one until
            // `ArenaPool::set_rank` learns where the slab landed.
            rank: AtomicUsize::new(UNRANKED_BASE + arena_idx),
            creation_pending: AtomicBool::new(true),
        }
    }

    /// This arena's address-order position — see [`Self::rank`].
    #[inline]
    fn rank(&self) -> usize {
        self.rank.load(Ordering::Relaxed)
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
                self.capacity.set(self.rank());
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
/// Per-format bitmap: bit `r` set ⇒ the arena at rank `r` has ≥1 free slot.
/// The pool's O(1) "which arena has capacity" index — it replaces the per-alloc
/// walk over every arena that was the pressure-regime `alloc` bottleneck
/// (`allocate_any` used to iterate the whole `tables` map, O(num_arenas), for
/// every one of the ~1M allocations a drain pass issues). Fully lock-free: `dec`
/// sets a bit on a full→non-full transition, alloc clears it on non-full→full
/// and reads it via find-first-set. Fixed-size so `dec` never contends a resize.
///
/// The bit position is the arena's **rank** — its region index, so its position
/// in span-address order — not its `arena_idx`. That is what makes
/// find-first-set return the leftmost arena with room, and it is the whole
/// mechanism behind address-ordered packing: see [`ArenaRefcounts::rank`].
#[derive(Debug)]
struct CapacityBitmap {
    words: Box<[AtomicU64]>,
}

/// Ranks reserved for arenas that hold a region: one per region of a 512 GiB
/// KV side, far past any card's reservation. Floored at 4096 so a small
/// `TARGET_ARENA_BYTES` cannot make this the binding limit.
const UNRANKED_BASE: usize = {
    let n = (512usize << 30) / TARGET_ARENA_BYTES;
    if n < 4096 {
        4096
    } else {
        n
    }
};

/// Every rank fits below this: `[0, UNRANKED_BASE)` for arenas whose region is
/// known, `[UNRANKED_BASE, 2 * UNRANKED_BASE)` for the ones still being carved.
/// `arena_idx` is bounded by the same region count (indices are recycled on
/// tombstone), so the second half is as wide as it needs to be.
const RANK_LIMIT: usize = 2 * UNRANKED_BASE;

impl CapacityBitmap {
    fn new() -> Self {
        let words = (0..RANK_LIMIT.div_ceil(64))
            .map(|_| AtomicU64::new(0))
            .collect();
        Self { words }
    }

    #[inline]
    fn set(&self, rank: usize) {
        if let Some(w) = self.words.get(rank >> 6) {
            w.fetch_or(1u64 << (rank & 63), Ordering::Release);
        }
    }

    #[inline]
    fn clear(&self, rank: usize) {
        if let Some(w) = self.words.get(rank >> 6) {
            w.fetch_and(!(1u64 << (rank & 63)), Ordering::Release);
        }
    }

    /// Lowest rank whose has-capacity bit is set, or `None` if all clear —
    /// i.e. the arena furthest left in the span that still has a free slot.
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
    /// `arena_idx → refcount table` — the **lookup** view. `RwLock`'d for
    /// register/release; reads are uncontended in steady state.
    ///
    /// This is what a gid resolves through, since a gid decodes to an
    /// `arena_idx`. It is deliberately *not* what the claim paths iterate: an
    /// arena index says nothing about where in the span its bytes are, so
    /// walking this in index order packs chunks into arenas sitting anywhere.
    /// That is [`Self::by_rank`]'s job.
    tables: RwLock<BTreeMap<usize, Arc<ArenaRefcounts>>>,
    /// The same tables keyed by **rank** — [`ArenaRefcounts::rank`], the arena's
    /// region index, so iteration order is span-address order.
    ///
    /// Two views of one set because the two questions are different: a gid
    /// decodes to an `arena_idx` and must resolve through `tables`, while every
    /// *claim* wants the leftmost arena with room and resolves through this.
    /// `capacity.first_set()` names a rank, so this is what turns that answer
    /// back into a table without a scan — one `BTreeMap` probe over a few
    /// hundred arenas, against the O(num_arenas) walk it replaces.
    ///
    /// Ranks are unique by construction (region index while carved,
    /// `UNRANKED_BASE + arena_idx` before), so the map holds exactly the arenas
    /// `tables` does and neither view can silently drop one.
    by_rank: RwLock<BTreeMap<usize, Arc<ArenaRefcounts>>>,
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
            by_rank: RwLock::new(BTreeMap::new()),
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
        {
            let mut by_rank = self.by_rank.write().unwrap();
            by_rank.insert(table.rank(), Arc::clone(&table));
        }
        self.total_arenas.fetch_add(1, Ordering::Relaxed);
        // A fresh arena is all free — mark it available. Ordered after both
        // inserts so a claimer that sees the bit also finds the table. The rank
        // is still the pre-region one, so the arena sorts behind every placed
        // arena until `set_rank` runs — which is what we want from an arena
        // whose slab does not exist yet.
        self.capacity.set(table.rank());
        table
    }

    /// Publish an arena's region, moving it into span-address order.
    ///
    /// Registration hands out an index before the slab is carved, so an arena
    /// is born unranked and sorts last. This runs the moment `create_arena` has
    /// its [`RegionHandle`](super::region_pool::RegionHandle) — and again after
    /// a relocation moves the arena to a lower region — so the claim path's
    /// find-first-set sees it at its real position in the span.
    ///
    /// Idempotent, and a no-op for an arena that is gone or already at `rank`.
    ///
    /// Moving the bit is deliberately not atomic with moving the map entry:
    /// nothing here can be, since `dec` sets bits lock-free from arbitrary
    /// threads. Both directions are benign. A bit left behind at the old rank
    /// resolves as a `by_rank` miss, which `allocate_any` already clears and
    /// steps past (the tombstoned-arena case). A bit lost at the new rank hides
    /// capacity, which the fallback resync rebuilds from `free_count`. Neither
    /// can strand a slot or hand one out twice.
    fn set_rank(&self, arena_idx: usize, rank: usize) {
        debug_assert!(
            rank < UNRANKED_BASE,
            "set_rank: {rank} is not a region rank"
        );
        let table = {
            let tables = self.tables.read().unwrap();
            match tables.get(&arena_idx) {
                Some(t) => Arc::clone(t),
                None => return,
            }
        };
        let old = table.rank();
        if old == rank {
            return;
        }
        {
            let mut by_rank = self.by_rank.write().unwrap();
            // Re-read under the write lock: two `set_rank` calls for the same
            // arena would otherwise both remove `old` and both insert, leaving
            // the loser's stale entry behind under a rank it no longer holds.
            let old = table.rank();
            if old == rank {
                return;
            }
            by_rank.remove(&old);
            table.rank.store(rank, Ordering::Relaxed);
            by_rank.insert(rank, table);
        }
        self.capacity.clear(old);
        // Only claim to have room if it does — an arena can be ranked after it
        // has already been filled by `allocate_run_in`.
        if self
            .tables
            .read()
            .unwrap()
            .get(&arena_idx)
            .is_some_and(|t| t.free_count() > 0)
        {
            self.capacity.set(rank);
        }
    }

    /// Claim `len` CONSECUTIVE slots in one arena of this pool, returning the
    /// first raw gid and the arena's refcount table. Only the never-used
    /// high-water tail of an arena can host a run (see
    /// [`ArenaRefcounts::try_claim_run`]); arenas whose tail is exhausted are
    /// skipped, and `None` means the caller must register a fresh arena.
    ///
    /// Walks `by_rank`, so it takes the leftmost arena in the span whose tail
    /// is long enough — the same address-order bias as [`Self::allocate_any`].
    fn allocate_run(&self, len: usize) -> Option<(i64, Arc<ArenaRefcounts>)> {
        let _gate = self.alloc_gate.lock().unwrap();
        let stride = GID_STRIDE;
        let by_rank = self.by_rank.read().unwrap();
        for table in by_rank.values() {
            if !table.run_fits(len) {
                continue;
            }
            if let Some(first) = table.try_claim_run(len) {
                if table.is_full() {
                    self.capacity.clear(table.rank());
                }
                return Some(((table.arena_idx * stride + first) as i64, Arc::clone(table)));
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
            self.capacity.clear(table.rank());
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
        let by_rank = self.by_rank.read().unwrap();
        // Fast path: the capacity bitmap points at the LEFTMOST arena in the
        // span with a free slot — find-first-set over ranks, and a rank is a
        // region index (see `ArenaRefcounts::rank`). That is what packs chunks
        // toward the bottom of the reservation so the arenas above drain and
        // give their regions back, and it skips the full-arena prefix that made
        // this an O(num_arenas) walk per alloc at pressure. A set bit can be
        // stale (the arena filled, or moved rank, since it was set) → the claim
        // fails → clear it and try the next set bit.
        while let Some(rank) = self.capacity.first_set() {
            match by_rank.get(&rank) {
                Some(table) => {
                    if let Some(chunk_idx) = table.try_claim_one() {
                        if table.is_full() {
                            self.capacity.clear(rank);
                        }
                        return Some((
                            (table.arena_idx * stride + chunk_idx) as i64,
                            Arc::clone(table),
                        ));
                    }
                    // Stale set bit — arena is full. Clear and try the next.
                    self.capacity.clear(rank);
                }
                // Bit set at a rank nobody holds — a tombstoned arena, or one
                // that `set_rank` moved out from under a concurrent `dec`.
                None => self.capacity.clear(rank),
            }
        }
        // Fallback: the bitmap says every arena is full. That's authoritative
        // *unless* a bit was over-cleared (an alloc's fill-clear raced a dec's
        // set, or a `set_rank` cleared the old position after the arena's last
        // drop set it), which can only hide capacity, never invent fullness.
        // Rebuild the whole bitmap once from the authoritative `free_count` —
        // recovering every over-cleared bit in a single pass — then retry the
        // fast path. We hold `alloc_gate`, so no concurrent claim can consume
        // the recovered capacity before we do; concurrent drops only add more.
        let mut recovered = false;
        for (&rank, table) in by_rank.iter() {
            if table.free_count() > 0 {
                self.capacity.set(rank);
                recovered = true;
            }
        }
        if recovered {
            if let Some(rank) = self.capacity.first_set() {
                if let Some(table) = by_rank.get(&rank) {
                    if let Some(chunk_idx) = table.try_claim_one() {
                        if table.is_full() {
                            self.capacity.clear(rank);
                        }
                        return Some((
                            (table.arena_idx * stride + chunk_idx) as i64,
                            Arc::clone(table),
                        ));
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
        // Leftmost-first walk in span-address order, draining each arena before
        // moving up — one read lock, no index collect + sort. The bitmap is no
        // help here: a bulk claim wants every arena in order, not the first.
        let stride = GID_STRIDE;
        let by_rank = self.by_rank.read().unwrap();
        for table in by_rank.values() {
            if out.len() == n {
                break;
            }
            let base = (table.arena_idx * stride) as i64;
            // Drain as many as we can from this arena.
            while out.len() < n {
                match table.try_claim_one() {
                    Some(chunk_idx) => {
                        out.push((base + chunk_idx as i64, Arc::clone(table)));
                    }
                    None => break,
                }
            }
            if table.is_full() {
                self.capacity.clear(table.rank());
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
        let removed = tables.remove(&candidate);
        drop(tables);
        self.total_arenas.fetch_sub(1, Ordering::Relaxed);
        // Arena gone — drop it from the ordered view and clear its capacity bit
        // so `allocate_any` doesn't chase a dangling rank (it self-heals via the
        // `by_rank.get` miss anyway).
        if let Some(t) = removed {
            let rank = t.rank();
            self.by_rank.write().unwrap().remove(&rank);
            self.capacity.clear(rank);
        }
        Some(candidate)
    }

    /// Force-remove an arena's table regardless of whether it's empty.
    /// Used by the legacy `release_arena` path after a manual gid drain.
    fn force_release(&self, arena_idx: usize) {
        let mut tables = self.tables.write().unwrap();
        let removed = tables.remove(&arena_idx);
        drop(tables);
        if let Some(t) = removed {
            self.total_arenas.fetch_sub(1, Ordering::Relaxed);
            let rank = t.rank();
            self.by_rank.write().unwrap().remove(&rank);
            self.capacity.clear(rank);
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
    /// Recycled arena indices from tombstoned arenas, **lowest first**, so the
    /// live index space stays packed against zero.
    ///
    /// # What this order is and is not for
    ///
    /// It is **not** what makes chunk allocation pack low. That was the original
    /// reasoning — that both sides handing out their lowest free unit would make
    /// the k-th arena by index the k-th by address, so packing by index *would
    /// be* packing by address — and it does not hold. An arena's region comes
    /// from a heap shared with the DeltaNet recurrent stores and the wave
    /// transient tier, so the two sequences are drawn from different pools and
    /// drift apart the moment another tenant takes a region. Index order is
    /// simply not address order, however either list is ordered.
    ///
    /// Packing low is [`ArenaRefcounts::rank`]'s job: the capacity bitmap is
    /// indexed by *region*, so find-first-set names the leftmost arena in the
    /// span directly, and nothing has to be inferred from the index at all.
    ///
    /// What this order still buys is **density**. `arena_idx` sizes
    /// `arena_registry`, and it is also the rank an arena carries in the window
    /// between registration and its slab being carved, which has to stay inside
    /// the bitmap's reserved upper half (`UNRANKED_BASE + arena_idx`). FIFO
    /// recycling let indices drift upward without bound under churn — an index
    /// freed long ago sat at the back of the queue while fresh ones climbed past
    /// it. Lowest-first keeps the live set compact.
    ///
    /// A heap rather than a sorted scan because this is arena creation, not
    /// chunk allocation: it runs on the order of hundreds of times a run
    /// against millions of claims, and the hot path — the capacity bitmap — is
    /// untouched.
    free_arenas: BinaryHeap<Reverse<usize>>,
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
                    free_arenas: BinaryHeap::with_capacity(16),
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
            let arena_idx = state
                .free_arenas
                .pop()
                .map(|Reverse(i)| i)
                .unwrap_or_else(|| {
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
    /// Serves the arena furthest **left in the span** that has room — the
    /// capacity bitmap is indexed by region, so find-first-set answers that in
    /// O(1) (see [`ArenaRefcounts::rank`]). Returns `None` if no arena of this
    /// format has free capacity — caller should `register_arena` and retry.
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
        state.free_arenas.push(Reverse(arena_idx));
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

    /// Tell the pool which region an arena's slab landed in, so claims can bias
    /// toward the arena furthest left in the span.
    ///
    /// Registration assigns an index before the slab exists, so an arena starts
    /// out sorting behind every placed one. Call this as soon as the region is
    /// known — and again whenever the arena is relocated to a different region —
    /// or the arena keeps drawing chunks as a last resort no matter how low in
    /// the span it actually sits.
    ///
    /// See [`ArenaRefcounts::rank`] for why the region index is the right key.
    ///
    /// The format comes from the registry rather than the caller: relocation
    /// works from an arena index alone, and reading the one authoritative
    /// mapping is better than asking two callers to agree on it.
    pub fn set_arena_rank(&self, arena_idx: usize, region_idx: usize) {
        let key = {
            let state = self.inner.metadata.lock().unwrap();
            state.arena_registry.get(arena_idx).and_then(|k| *k)
        };
        if let Some(pool) = key.and_then(|k| self.inner.pools.get(&k)) {
            pool.set_rank(arena_idx, region_idx);
        }
    }

    /// Remove from `free_arenas` any indices >= `threshold`.
    pub fn drain_free_arenas_above(&self, threshold: usize) {
        let mut state = self.inner.metadata.lock().unwrap();
        state.free_arenas.retain(|&Reverse(idx)| idx < threshold);
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
                state.free_arenas.push(Reverse(arena_idx));
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

    /// **Chunks go to the arena lowest in the SPAN, not the lowest-numbered
    /// arena.**
    ///
    /// These are different orders and nothing keeps them together: arena
    /// indices come from the pool's own recycled free list, regions from
    /// `claim_region` — a heap shared with the DeltaNet recurrent stores and
    /// the wave transient tier. An index says nothing about where the bytes
    /// sit.
    ///
    /// It has to be the address order, because the quantity the packing exists
    /// to move is `live_watermark` — the highest occupied region — and that is
    /// what sets both the weight zone's growth headroom and the wave tier's
    /// budget. Packing by index concentrated chunks into arenas scattered
    /// anywhere in the span, so the top never drained and the watermark stayed
    /// where its high-water mark had left it.
    ///
    /// The arenas here are registered in ascending index order and ranked in
    /// the *opposite* order, so an allocator that still keyed on the index
    /// would answer every assertion below backwards.
    #[test]
    fn claims_go_to_the_leftmost_arena_in_the_span() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();

        let low_idx = pool.register_arena(key);
        let mid_idx = pool.register_arena(key);
        let high_idx = pool.register_arena(key);

        // Index order and address order deliberately disagree: the
        // highest-numbered arena sits lowest in the span.
        pool.set_arena_rank(low_idx, 40);
        pool.set_arena_rank(mid_idx, 20);
        pool.set_arena_rank(high_idx, 4);

        // Drain in order and record which arena served each chunk. Three
        // arenas' worth, so every one is visited. The gids are held (not
        // dropped) so each claim sees the fill left by the one before.
        let mut held: Vec<ChunkGid> = Vec::new();
        let mut served: Vec<usize> = Vec::new();
        for _ in 0..(cap * 3) {
            let gid = pool.allocate_for(key).expect("pool has room");
            served.push(gid.arena_idx());
            held.push(gid);
        }

        // Region 4 first, then 20, then 40 — address order throughout.
        assert!(
            served[..cap].iter().all(|&a| a == high_idx),
            "the first {cap} chunks must fill region 4 (arena {high_idx}), got {:?}",
            &served[..cap.min(served.len())],
        );
        assert!(
            served[cap..cap * 2].iter().all(|&a| a == mid_idx),
            "the next {cap} must fill region 20 (arena {mid_idx})",
        );
        assert!(
            served[cap * 2..].iter().all(|&a| a == low_idx),
            "the last {cap} must fill region 40 (arena {low_idx})",
        );
    }

    /// **An arena whose slab has not been carved yet is the last resort.**
    ///
    /// Registration hands out an index before `create_arena` claims a region,
    /// so for that window the pool cannot know where the arena will land. It
    /// must not guess: ranking an unplaced arena low would send claims to it
    /// ahead of arenas known to be at the bottom of the span, which is the
    /// scattering this ordering exists to prevent. Ranking it past every real
    /// region gets the safe answer, and `set_arena_rank` moves it the moment
    /// the region is known.
    ///
    /// The unplaced arena must still be *reachable* — an arena missing from the
    /// claim path is worse than one in the wrong place, because the allocator
    /// would register a fresh arena for every chunk it could not place.
    #[test]
    fn an_unplaced_arena_sorts_behind_every_placed_one() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();

        // Registered FIRST, so a lowest-index policy would prefer it.
        let unplaced = pool.register_arena(key);
        let placed = pool.register_arena(key);
        pool.set_arena_rank(placed, 900);

        let mut held: Vec<ChunkGid> = Vec::new();
        let first = pool.allocate_for(key).expect("pool has room");
        assert_eq!(
            first.arena_idx(),
            placed,
            "a placed arena at region 900 still beats an arena with no region",
        );
        held.push(first);

        // Fill the placed arena; the unplaced one then serves, rather than the
        // pool reporting itself full.
        for _ in 1..cap {
            let gid = pool.allocate_for(key).expect("placed arena has room");
            assert_eq!(gid.arena_idx(), placed);
            held.push(gid);
        }
        let spill = pool
            .allocate_for(key)
            .expect("the unplaced arena must still be reachable");
        assert_eq!(spill.arena_idx(), unplaced);
        held.push(spill);
    }

    /// **Relocating an arena re-points the claim path at its new address.**
    ///
    /// Compaction moves an arena's bytes down the span; if its rank stayed
    /// where it was, allocation would go on filling whatever sits above it and
    /// undo the move on the next wave. This is the half that makes compaction
    /// and allocation pull the same way.
    #[test]
    fn relocating_an_arena_moves_it_to_the_front_of_the_claim_order() {
        let pool = ChunkGidPool::new();
        let key = float_key();

        let a = pool.register_arena(key);
        let b = pool.register_arena(key);
        pool.set_arena_rank(a, 30);
        pool.set_arena_rank(b, 12);

        let mut held: Vec<ChunkGid> = Vec::new();
        let before = pool.allocate_for(key).expect("pool has room");
        assert_eq!(before.arena_idx(), b, "region 12 is the leftmost");
        held.push(before);

        // `a` is compacted down past `b`.
        pool.set_arena_rank(a, 3);

        let after = pool.allocate_for(key).expect("pool has room");
        assert_eq!(
            after.arena_idx(),
            a,
            "after relocation to region 3, arena {a} is the leftmost",
        );
        held.push(after);

        // Re-ranking is idempotent and leaves no stale position behind: the
        // answer does not change when the same rank is published twice.
        pool.set_arena_rank(a, 3);
        let again = pool.allocate_for(key).expect("pool has room");
        assert_eq!(again.arena_idx(), a);
        held.push(again);
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
