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
//! ### Placement order
//!
//! Every choice of arena — which takes the next slot, which empty one is
//! released — goes by the arena's **place**: its region's position in the
//! device reservation, lowest address first ([`ArenaPool::set_rank`]). An
//! `arena_idx` is only the gid namespace slot, recycled whenever an arena is
//! tombstoned, so it says nothing about where the arena sits; ordering by it
//! put new chunks and the kept reserve wherever recycling had left the index,
//! including the region at the top of the span. An arena with no region (a
//! warm-tier slab, or one whose region is not yet claimed) sorts after every
//! placed arena, by index.
//!
//! ### Allocation path
//!
//! `ArenaPool::allocate_n` iterates arenas in place order (skipping full
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
use std::cmp::Reverse;
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

use super::arena::{ArenaKey, ArenaKind, RECORD_STRIDES};
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
    /// compose. Read by [`ArenaRefcounts::occupied_slots`] to enumerate live
    /// slots for a compaction census.
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
    /// This arena's place in its pool's placement order — its region rank once
    /// [`ArenaPool::set_rank`] records it, [`unranked_place`] until then — and
    /// so the bit it owns in `capacity`.
    place: AtomicUsize,
    /// Creation window guard: `true` from registration until the arena hands out
    /// its FIRST gid, or until its creator closes it explicitly with
    /// `ChunkGidPool::finish_creation` — which every creator that may not claim does
    /// (a stamp after its by-name claim, a compaction's provisioned arenas once built). `register_arena` releases the metadata lock before
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
            place: AtomicUsize::new(unranked_place(arena_idx)),
            creation_pending: AtomicBool::new(true),
        }
    }

    /// This arena's place in its pool's placement order.
    #[inline]
    fn place(&self) -> usize {
        self.place.load(Ordering::Acquire)
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

    /// Occupied slot indices, ascending — the census a compaction plans from.
    ///
    /// Walks the [`Self::occupancy`] bitmap a word at a time and peels set bits
    /// with `trailing_zeros`, so the cost is the bitmap's words plus the live
    /// count rather than the arena's capacity. That matters at the bottom of the
    /// ladder, where one arena holds 52,428 slots and a per-slot loop would
    /// dominate a pass that has only a handful of chunks to move.
    ///
    /// **A snapshot, not a lock.** Each word is read independently, so a claim or
    /// a drop landing mid-walk may or may not be seen. The caller compensates
    /// structurally rather than by locking: a compaction runs inside the arena
    /// window with no forward in flight, and the destination claim
    /// (`allocate_from_arena`) is what finally refuses a slot the census believed
    /// free. A stale *source* is harmless — the move is simply not worth making.
    pub(crate) fn occupied_slots(&self) -> Vec<u32> {
        let live = self.live.load(Ordering::Acquire);
        let mut out = Vec::with_capacity(live);
        for (w, word) in self.occupancy.iter().enumerate() {
            let mut bits = word.load(Ordering::Acquire);
            while bits != 0 {
                let b = bits.trailing_zeros();
                bits &= bits - 1;
                let slot = w * 64 + b as usize;
                // The last word is padded past `arena_chunks`; a padding bit is
                // never set, but bounding here means a future change to the
                // padding cannot hand a caller a slot the arena does not have.
                if slot < self.arena_chunks {
                    out.push(slot as u32);
                }
            }
        }
        out
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
            #[cfg(feature = "tensor-assert")]
            super::release_watch::on_release((self.arena_idx * GID_STRIDE + chunk_idx) as i64);
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
                self.capacity.set(self.place());
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
/// Places a ranked arena can take — one per region of the largest span the pool
/// is sized for: a 512 GiB VRAM ceiling over the arena size, far past any single
/// card. Region ranks and `arena_idx` (VRAM-bounded, recycled on tombstone) both
/// stay below it.
const RANKED_PLACES: usize = {
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
const fn unranked_place(arena_idx: usize) -> usize {
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
struct CapacityBitmap {
    words: Box<[AtomicU64]>,
}

impl CapacityBitmap {
    fn new() -> Self {
        let words = (0..(2 * RANKED_PLACES).div_ceil(64))
            .map(|_| AtomicU64::new(0))
            .collect();
        Self { words }
    }

    #[inline]
    fn set(&self, place: usize) {
        if let Some(w) = self.words.get(place >> 6) {
            w.fetch_or(1u64 << (place & 63), Ordering::Release);
        }
    }

    #[inline]
    fn clear(&self, place: usize) {
        if let Some(w) = self.words.get(place >> 6) {
            w.fetch_and(!(1u64 << (place & 63)), Ordering::Release);
        }
    }

    /// Lowest place whose has-capacity bit is set, or `None` if all clear.
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
    tables: RwLock<BTreeMap<usize, Arc<ArenaRefcounts>>>,
    /// `place → arena_idx` for every arena in `tables`: the placement order the
    /// allocators walk and the empty sweep chooses by, lowest in the span
    /// first, kept so neither rebuilds and sorts it per claim. Taken after
    /// `tables`, never before.
    order: RwLock<BTreeMap<usize, usize>>,
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
    /// Bit `p` set ⇒ the arena at place `p` has ≥1 free slot. `allocate_any`
    /// finds the lowest such arena via find-first-set instead of walking every
    /// arena.
    capacity: Arc<CapacityBitmap>,
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
    run_floors: [AtomicUsize; RUN_FLOOR_LENS],
}

/// Run lengths with a floor of their own in [`ArenaPool::run_floors`]; a longer
/// run starts from the last one's, which no longer run can be served below.
const RUN_FLOOR_LENS: usize = 33;

impl ArenaPool {
    fn new(class: SizeClass) -> Self {
        Self::with_chunks(class.chunks_per_region())
    }

    /// A pool whose arenas hold `arena_chunks` slots.
    ///
    /// Split out because a record arena's capacity comes from its stride, not from its
    /// size class — `ArenaKey::chunks()` is the authority for both kinds, and taking it
    /// from the class for a record pool would size every refcount table wrongly.
    fn with_chunks(arena_chunks: usize) -> Self {
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
    fn run_floor(&self, len: usize) -> &AtomicUsize {
        &self.run_floors[len.min(RUN_FLOOR_LENS - 1)]
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
    fn set_rank(&self, arena_idx: usize, rank: usize) {
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
    fn remove_table(
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
    fn allocate_run(&self, len: usize) -> Option<(i64, Arc<ArenaRefcounts>)> {
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
    fn allocate_run_in(&self, arena_idx: usize, len: usize) -> Option<(i64, Arc<ArenaRefcounts>)> {
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
    fn at_place<'t>(
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

    fn allocate_any(&self) -> Option<(i64, Arc<ArenaRefcounts>)> {
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
    fn allocate_n(&self, n: usize) -> Vec<(i64, Arc<ArenaRefcounts>)> {
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
    ///
    /// **A reserve class keeps its target free** (`reserve`, in slots, at
    /// least one arena): an empty arena is released only while the class would
    /// still hold its target free without it — and, above the one-arena floor,
    /// one arena more, a hysteresis band so a target moving by a few slots
    /// neither creates nor releases.
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
    /// short of its target, the next gap owes it an arena, and a claim takes
    /// the lowest free region — the reserve moves down instead of pinning the
    /// frontier. On WhiteWorld (2026-11-10) a 1,024 B class kept an empty arena
    /// on region 487 as its reserve, with ~430 free regions (6.7 GB) below it
    /// and the expert zone 1 GB under its floor.
    fn try_tombstone(
        &self,
        protected_arenas: &AHashSet<usize>,
        reserve: Option<usize>,
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
        if let Some(target) = reserve {
            let free: usize = tables
                .values()
                .map(|t| self.arena_chunks.saturating_sub(t.live_count()))
                .sum();
            let above_a_hole = matches!(
                (highest_empty, lowest_hole),
                (Some((place, _)), Some(hole)) if place < RANKED_PLACES && hole < place
            );
            if free.saturating_sub(self.arena_chunks) < keep_free(target, self.arena_chunks)
                && !above_a_hole
            {
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
    fn tombstone_if_empty(&self, arena_idx: usize) -> bool {
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
    fn force_release(&self, arena_idx: usize) {
        self.remove_table(&mut self.tables.write().unwrap(), arena_idx);
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

    /// Add this pool's arenas that hold no chunk to `census`, each counted once more
    /// under the reason [`Self::try_tombstone`] would pass over it, if it has one.
    fn count_empty(&self, protected_arenas: &AHashSet<usize>, census: &mut EmptyCensus) {
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
    fn free_len(&self) -> usize {
        self.total_free()
    }
}

/// One key's arenas and how full they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyOccupancy {
    pub key: ArenaKey,
    /// Arenas registered for the key.
    pub arenas: usize,
    /// Slots holding a chunk, across those arenas.
    pub live: usize,
    /// Slots in one arena of the key.
    pub arena_chunks: usize,
}

/// The arenas of a pool that hold no chunk, and why the empty sweep would keep any
/// of them. An arena can be counted under both reasons.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmptyCensus {
    /// Arenas with no live chunk.
    pub empty: usize,
    /// Of those, arenas still inside their creation window.
    pub in_creation: usize,
    /// Of those, arenas protected from the sweep.
    pub protected: usize,
}

/// The free slots a reserve class keeps beyond the arena a sweep would
/// release: its target, never below one arena (`arena`), with one arena more
/// of hysteresis once the target is above that floor.
fn keep_free(target: usize, arena: usize) -> usize {
    if target > arena {
        target + arena
    } else {
        arena
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
    /// Arena indices protected from the empty sweep: arenas created ahead of
    /// a demand that has not filled them yet (`docs/kv_drain_under_pressure.md`
    /// §2). Lifted once the demand arrives or is abandoned.
    protected_arenas: AHashSet<usize>,
    /// The classes that keep a **reserve**, each with its target in free
    /// slots (`docs/kv_drain_under_pressure.md` §2a): never less than one
    /// arena's worth, grown with the native backlog the migrate has yet to
    /// quantize. The empty sweep never takes a class below its target. A
    /// hot→warm migrate into one of them finds its destination without
    /// creating an arena, which it cannot do while a wave is open.
    reserve_keys: AHashMap<ArenaKey, usize>,
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
                    reserve_keys: AHashMap::with_capacity(16),
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

    /// Keep a reserve for `key`, at its floor of one arena's worth of free
    /// slots until a target is set ([`Self::set_reserve_target`]).
    pub fn keep_reserve(&self, key: ArenaKey) {
        let mut state = self.inner.metadata.lock().unwrap();
        state.reserve_keys.entry(key).or_insert(0);
    }

    /// Set `key`'s reserve target in free slots; below one arena it is held
    /// at that floor. Keeps a reserve for `key` if it kept none.
    pub fn set_reserve_target(&self, key: ArenaKey, slots: usize) {
        let mut state = self.inner.metadata.lock().unwrap();
        state.reserve_keys.insert(key, slots);
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
        state.reserve_keys.keys().copied().collect()
    }

    /// The free slots `key`'s reserve holds: its target, never below one
    /// arena's worth; `None` for a class that keeps no reserve.
    pub fn reserve_target(&self, key: ArenaKey) -> Option<usize> {
        let state = self.inner.metadata.lock().unwrap();
        let arena = key.chunks();
        state.reserve_keys.get(&key).map(|&t| t.max(arena))
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
    fn run_floor_for(&self, key: ArenaKey, len: usize) -> usize {
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
        let reserve = state.reserve_keys.get(&key).copied();
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

#[cfg(test)]
mod tests {
    use super::*;
    fn float_key() -> ArenaKey {
        ArenaKey::new(TEST_CLASS, ArenaLocation::Gpu)
    }

    fn test_arena_chunks() -> usize {
        TEST_CLASS.chunks_per_region()
    }

    /// An arena of `key` registered and past its creation window, empty.
    fn standing_arena(pool: &ChunkGidPool, key: ArenaKey) -> usize {
        let idx = pool.register_arena(key);
        pool.finish_creation(key, idx);
        idx
    }

    /// **A reserve class keeps one arena's worth free, the lowest standing**
    /// (`docs/kv_drain_under_pressure.md` §2): of two empty arenas the sweep
    /// takes the higher and keeps the lower, then takes nothing more; a class
    /// with no reserve is swept bare.
    #[test]
    fn a_reserve_class_keeps_one_arenas_worth_free_at_its_lowest() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let other = ArenaKey::new(SizeClass::at(1), ArenaLocation::Gpu);
        pool.keep_reserve(key);
        let low = standing_arena(&pool, key);
        let high = standing_arena(&pool, key);
        assert!(low < high);
        assert_eq!(
            pool.next_tombstone(key, None),
            Some(high),
            "the higher goes first"
        );
        pool.recycle_arena_index(high);
        assert_eq!(pool.next_tombstone(key, None), None, "the reserve stays");
        assert_eq!(pool.free_slots(key), test_arena_chunks());

        let a = standing_arena(&pool, other);
        let b = standing_arena(&pool, other);
        assert_eq!(pool.next_tombstone(other, None), Some(a.max(b)));
        assert_eq!(
            pool.next_tombstone(other, None),
            Some(a.min(b)),
            "no reserve, swept bare"
        );
    }

    /// Two standing arenas of `key` whose indices run **opposite** to their
    /// places in the span: `(high, low)`, `high` registered first (the lower
    /// index) and carved from region 9, `low` second and from region 2.
    fn reversed_pair(pool: &ChunkGidPool, key: ArenaKey) -> (usize, usize) {
        let high = standing_arena(pool, key);
        let low = standing_arena(pool, key);
        assert!(high < low, "the lower index is the higher place");
        pool.set_arena_rank(key, high, 9);
        pool.set_arena_rank(key, low, 2);
        (high, low)
    }

    /// **The reserve that stays is the lowest in the span, whatever the
    /// indices say**: of two empty arenas the sweep releases the one at the
    /// greater offset into the reservation. Chosen by index, it released the
    /// lower arena and kept the higher one standing on the frontier — region
    /// 3127 on BlackWorld (2026-11-08), for hours.
    #[test]
    fn the_reserve_keeps_the_lowest_placed_arena_whatever_its_index() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.keep_reserve(key);
        let (high, low) = reversed_pair(&pool, key);
        assert_eq!(
            pool.next_tombstone(key, None),
            Some(high),
            "the arena higher in the span goes"
        );
        pool.recycle_arena_index(high);
        assert_eq!(pool.next_tombstone(key, None), None, "the lower one stays");
        assert_eq!(pool.pool_arena_load(key)[0].0, low);
    }

    /// **A reserve does not hold the top while a hole lies below it.** The one
    /// empty arena a reserve keeps is released when a free region stands below
    /// it, so the reserve is made again at the lowest hole; with no hole below,
    /// or one above, it stays.
    #[test]
    fn a_reserve_above_a_hole_is_released_to_move_down() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.keep_reserve(key);
        let only = standing_arena(&pool, key);
        pool.set_arena_rank(key, only, 487);
        assert_eq!(pool.next_tombstone(key, None), None, "no regions: it stays");
        assert_eq!(
            pool.next_tombstone(key, Some(600)),
            None,
            "the only hole is above it: it stays"
        );
        assert_eq!(
            pool.next_tombstone(key, Some(57)),
            Some(only),
            "a hole below: it goes, to be made again there"
        );
    }

    /// **A class with no reserve is swept from the top of the span down.**
    #[test]
    fn an_unreserved_class_is_swept_from_the_highest_place() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let (high, low) = reversed_pair(&pool, key);
        assert_eq!(pool.next_tombstone(key, None), Some(high));
        assert_eq!(pool.next_tombstone(key, None), Some(low));
    }

    /// **A slot goes to the arena lowest in the span, whatever its index** —
    /// a single claim, a bulk claim and a run alike. Filling by index put new
    /// chunks in the high arenas compaction had just emptied, and the two moved
    /// the same chunks back and forth every few seconds.
    #[test]
    fn a_claim_takes_the_lowest_placed_arena_whatever_its_index() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let (high, low) = reversed_pair(&pool, key);

        let one = pool.allocate_for(key).expect("a slot");
        assert_eq!(one.arena_idx(), low, "single claim");
        let bulk = pool.allocate_n_for(key, 3);
        assert!(bulk.iter().all(|g| g.arena_idx() == low), "bulk claim");
        let run = pool.allocate_run_for(key, 2).expect("a run");
        assert!(run.iter().all(|g| g.arena_idx() == low), "run claim");
        assert!(pool.run_would_fit(key, 2));

        // Fill the low arena: only then does the high one serve.
        let cap = test_arena_chunks();
        let rest = pool.allocate_n_for(key, cap - 6);
        assert!(rest.iter().all(|g| g.arena_idx() == low));
        assert_eq!(pool.allocate_for(key).unwrap().arena_idx(), high);
    }

    /// **A freed slot in a full low arena brings it back first**: the drop sets
    /// the arena's capacity at its place, not at its index.
    #[test]
    fn a_drop_in_the_low_arena_serves_the_next_claim() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let (high, low) = reversed_pair(&pool, key);
        let cap = test_arena_chunks();
        let mut filled = pool.allocate_n_for(key, cap);
        assert!(filled.iter().all(|g| g.arena_idx() == low));
        let in_high = pool.allocate_for(key).unwrap();
        assert_eq!(in_high.arena_idx(), high);
        filled.pop();
        assert_eq!(pool.allocate_for(key).unwrap().arena_idx(), low);
    }

    /// **A placed arena comes before every unplaced one**, and a recycled index
    /// starts unplaced: the rank went with the table its tombstone dropped, so
    /// the index does not carry its old region's place onto a new one.
    #[test]
    fn a_recycled_index_does_not_keep_its_old_place() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let first = standing_arena(&pool, key);
        pool.set_arena_rank(key, first, 0);
        assert_eq!(pool.next_tombstone(key, None), Some(first));
        pool.recycle_arena_index(first);
        let reused = standing_arena(&pool, key);
        assert_eq!(reused, first, "the index is recycled");
        let placed = standing_arena(&pool, key);
        pool.set_arena_rank(key, placed, 7);
        assert_eq!(
            pool.allocate_for(key).unwrap().arena_idx(),
            placed,
            "the placed arena serves before the unplaced recycled one"
        );
    }

    /// **An empty arena of a reserve class goes only while the class keeps a
    /// whole arena free without it**: one slot live in the other arena leaves
    /// one slot short, so the empty arena is the reserve and stays.
    #[test]
    fn an_empty_arena_stays_while_the_class_would_fall_short_without_it() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.keep_reserve(key);
        let _first = standing_arena(&pool, key);
        let held = pool.allocate_for(key).expect("a slot in the first arena");
        let _second = standing_arena(&pool, key);
        assert_eq!(pool.free_slots(key), 2 * test_arena_chunks() - 1);
        assert_eq!(
            pool.next_tombstone(key, None),
            None,
            "one slot short of a reserve without it"
        );
        drop(held);
        assert!(
            pool.next_tombstone(key, None).is_some(),
            "both empty: one may go"
        );
        assert_eq!(pool.next_tombstone(key, None), None, "and one stays");
    }

    /// **A target above the floor holds that many slots free, with one arena
    /// of hysteresis**: at a three-arena target, five empty arenas sweep down
    /// to four — the target plus the band — and no lower; at the floor the
    /// band is gone and the class sweeps to one.
    #[test]
    fn a_reserve_target_holds_its_slots_free_with_one_arena_of_hysteresis() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let arena = test_arena_chunks();
        pool.set_reserve_target(key, 3 * arena);
        assert_eq!(pool.reserve_target(key), Some(3 * arena));
        for _ in 0..5 {
            standing_arena(&pool, key);
        }
        let first = pool
            .next_tombstone(key, None)
            .expect("five over a band of four");
        pool.recycle_arena_index(first);
        assert_eq!(
            pool.next_tombstone(key, None),
            None,
            "target plus the band stays"
        );
        assert_eq!(pool.free_slots(key), 4 * arena);

        pool.set_reserve_target(key, 0);
        assert_eq!(pool.reserve_target(key), Some(arena), "held at the floor");
        for _ in 0..3 {
            let idx = pool.next_tombstone(key, None).expect("above the floor");
            pool.recycle_arena_index(idx);
        }
        assert_eq!(pool.next_tombstone(key, None), None, "one arena stays");
        assert_eq!(pool.free_slots(key), arena);
    }

    /// **No reserve is kept once cleared**, for a backing that no longer
    /// writes the classes its reserves were named at.
    #[test]
    fn a_cleared_reserve_is_swept_like_any_class() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        pool.keep_reserve(key);
        let only = standing_arena(&pool, key);
        assert_eq!(pool.next_tombstone(key, None), None);
        pool.clear_reserves();
        assert!(pool.reserve_keys().is_empty());
        assert_eq!(pool.next_tombstone(key, None), Some(only));
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
            pool.next_tombstone(key, None),
            None,
            "an arena inside its creation window must not be tombstoned"
        );

        // The creator finishes without ever claiming a chunk — exactly what
        // pre-creating a slab for a class that asked for one looks like.
        pool.finish_creation(key, idx);

        assert_eq!(
            pool.next_tombstone(key, None),
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
            pool.next_tombstone(key, None),
            None,
            "an occupied arena is not reclaimable"
        );

        drop(gid);
        assert_eq!(
            pool.next_tombstone(key, None),
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

    /// Runs go to the lowest arena that still has a tail, and the arenas whose
    /// tails are gone are passed over — the placement the walk always gave, now
    /// without visiting every exhausted arena on every claim.
    #[test]
    fn a_run_lands_in_the_lowest_arena_with_a_tail() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();
        let a = pool.register_arena(key);
        let b = pool.register_arena(key);
        let c = pool.register_arena(key);

        // Arena `a` loses its whole tail; `b` all but two slots.
        let fill_a = pool.allocate_run_for(key, cap).unwrap();
        assert!(fill_a.iter().all(|g| g.arena_idx() == a));
        let fill_b = pool.allocate_run_for(key, cap - 2).unwrap();
        assert!(fill_b.iter().all(|g| g.arena_idx() == b));

        // The two that remain in `b` are taken before `c` is touched.
        let r1 = pool.allocate_run_for(key, 2).unwrap();
        assert!(r1.iter().all(|g| g.arena_idx() == b));
        let r2 = pool.allocate_run_for(key, 2).unwrap();
        assert!(r2.iter().all(|g| g.arena_idx() == c));

        // A new arena registered above does not displace `c`.
        pool.register_arena(key);
        let r3 = pool.allocate_run_for(key, 2).unwrap();
        assert!(r3.iter().all(|g| g.arena_idx() == c));
    }

    /// **A tombstoned index is not reissued until its storage is gone.** The pool
    /// and the arena storage release an arena in two steps, and a registration in
    /// between used to take the index while storage still held the old arena —
    /// which `ensure_arena_exists` then adopted, old stride and all, under the new
    /// key. Compaction found the result as a copy between slots of unequal extent:
    /// arena 398 holding 4096 B slots in a pool of 8192 B ones.
    #[test]
    fn a_tombstoned_index_waits_for_its_storage_release() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let other = ArenaKey::new(SizeClass::at(1), ArenaLocation::Gpu);
        let a = pool.register_arena(key);
        pool.finish_creation(key, a);
        assert_eq!(pool.next_tombstone(key, None), Some(a));
        let b = pool.register_arena(other);
        assert_ne!(
            b, a,
            "the index was reissued while its storage could still hold the old arena"
        );
        pool.recycle_arena_index(a);
        assert_eq!(
            pool.register_arena(other),
            a,
            "and is reissued once released"
        );
    }

    /// **A tail too short for a run moves that run length's floor past it, and
    /// only that length's.** Two arenas are left one slot short of empty; a run
    /// of two lands in the third and the length-2 walk starts there from now on,
    /// while a single slot still finds the lowest arena with one left. With one
    /// floor kept for "no tail at all", the one-slot tails held it at the first
    /// arena and every run claim walked the whole pool.
    #[test]
    fn a_short_tail_moves_only_its_run_lengths_floor() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();
        let a = pool.register_arena(key);
        let b = pool.register_arena(key);
        let c = pool.register_arena(key);

        let fill_a = pool.allocate_run_for(key, cap - 1).unwrap();
        assert!(fill_a.iter().all(|g| g.arena_idx() == a));
        let fill_b = pool.allocate_run_for(key, cap - 1).unwrap();
        assert!(fill_b.iter().all(|g| g.arena_idx() == b));

        // A floor is a place: these arenas carry no region, so theirs follow
        // their indices after every ranked place.
        let pair = pool.allocate_run_for(key, 2).unwrap();
        assert!(pair.iter().all(|g| g.arena_idx() == c));
        assert_eq!(pool.run_floor_for(key, 2), unranked_place(c));
        assert_eq!(pool.run_floor_for(key, 3), 0, "no length-3 walk ran");
        assert_eq!(pool.run_floor_for(key, 1), 0);

        let single = pool.allocate_run_for(key, 1).unwrap();
        assert_eq!(single[0].arena_idx(), a);
        let single = pool.allocate_run_for(key, 1).unwrap();
        assert_eq!(single[0].arena_idx(), b);
        assert_eq!(pool.run_floor_for(key, 1), unranked_place(b));
    }

    /// An arena that registers below the point the walk has passed is a fresh
    /// arena with a whole tail, and it is preferred again — lowest-first
    /// placement is what compaction packs against, and recycled indices are how
    /// a low arena comes back.
    #[test]
    fn a_recycled_low_arena_is_preferred_again_after_the_walk_has_passed_it() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let cap = test_arena_chunks();
        let a = pool.register_arena(key);
        let b = pool.register_arena(key);

        let fill_a = pool.allocate_run_for(key, cap).unwrap();
        let in_b = pool.allocate_run_for(key, 2).unwrap();
        assert!(in_b.iter().all(|g| g.arena_idx() == b));

        // `a` empties, its storage is released, and its index is recycled by the
        // next registration.
        drop(fill_a);
        assert_eq!(pool.next_tombstone(key, None), Some(a));
        pool.recycle_arena_index(a);
        assert_eq!(pool.register_arena(key), a);

        let run = pool.allocate_run_for(key, 2).unwrap();
        assert!(
            run.iter().all(|g| g.arena_idx() == a),
            "the recycled arena below the exhausted prefix has the lowest tail"
        );
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
            pool.next_tombstone(key, None),
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

    /// **The census counts every empty arena, and names what would keep each one from
    /// the sweep**: an open creation window, a protection, both, or nothing.
    #[test]
    fn the_empty_census_names_why_each_empty_arena_is_kept() {
        let pool = ChunkGidPool::new();
        let key = float_key();
        let used = pool.register_arena(key);
        let _live = pool.allocate_from_arena(key, used).unwrap();
        let _pending = pool.register_arena(key);
        let released = pool.register_arena(key);
        pool.finish_creation(key, released);
        let guarded = pool.register_arena(key);
        pool.finish_creation(key, guarded);
        pool.protect_arena(guarded);
        let both = pool.register_arena(key);
        pool.protect_arena(both);

        assert_eq!(
            pool.empty_census(),
            EmptyCensus {
                empty: 4,
                in_creation: 2,
                protected: 2,
            }
        );
        assert_eq!(pool.next_tombstone(key, None), Some(released));
        assert_eq!(pool.next_tombstone(key, None), None, "the rest are kept");
        assert_eq!(
            pool.occupancy(),
            [KeyOccupancy {
                key,
                arenas: 4,
                live: 1,
                arena_chunks: test_arena_chunks(),
            }]
        );
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
