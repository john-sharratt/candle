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
use candle::DType;
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
use crate::kv_cache::chunked::types::{
    arena_chunks_for_format, arena_gid_stride, TARGET_ARENA_BYTES,
};
use crate::kv_cache::{ArenaLocation, KvFormat, QuantFormat};

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
        self.creation_pending.store(false, Ordering::Release);
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

    /// Iterate over the raw gids currently allocated in this arena.
    /// Slow path used by diagnostics / live-gid queries — scans every
    /// slot. Only called from the CUDA-gated defrag path.
    #[cfg(feature = "cuda")]
    fn live_gids(&self) -> Vec<i64> {
        let base = (self.arena_idx * arena_gid_stride()) as i64;
        let mut out = Vec::with_capacity(self.live_count());
        for i in 0..self.arena_chunks {
            // `counts[i]` is now overlapped (refcount when occupied, free-list
            // link when free), so the occupancy bit is the free/occupied
            // discriminator — not `counts > 0`.
            if self.occupancy[i / 64].load(Ordering::Acquire) & (1u64 << (i % 64)) != 0 {
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
    fn new(format: KvFormat) -> Self {
        Self {
            tables: RwLock::new(BTreeMap::new()),
            total_arenas: AtomicUsize::new(0),
            total_live: Arc::new(AtomicUsize::new(0)),
            arena_chunks: arena_chunks_for_format(format),
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
    fn allocate_any(&self) -> Option<(i64, Arc<ArenaRefcounts>)> {
        // Serialize the claiming walk: only one thread scans this pool's
        // `counts` arrays at a time, so `try_claim_one` can probe occupancy
        // with a relaxed load (no per-slot locked RMW) without 128 threads
        // ping-ponging the same cache lines. Drops stay lock-free.
        let _gate = self.alloc_gate.lock().unwrap();
        let stride = arena_gid_stride();
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

/// GPU arena occupancy split by float vs quant format — the diagnostic the
/// compress-to-free relief rung is judged by. Compress shrinks the float side
/// (working-set arenas the persistence thread would quantize anyway) while the
/// quant side grows less, so a working rung shows float bytes dropping across a
/// pressure episode. `reserved` counts whole ~16 MiB slabs
/// ([`TARGET_ARENA_BYTES`]); `live` counts occupied chunk slots at the format's
/// per-chunk size. These are GidPool arena-slab quantities, distinct from the
/// CUDA stream-ordered pool's `reserved`/`used` (which include segment slack the
/// GidPool never sees) — report them on their own line, not as a partition of
/// the CUDA-pool gap.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuArenaFormatStats {
    pub float_arenas: usize,
    pub float_reserved_bytes: usize,
    pub float_live_bytes: usize,
    pub quant_arenas: usize,
    pub quant_reserved_bytes: usize,
    pub quant_live_bytes: usize,
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

    /// GPU arena occupancy split float vs quant. Reads the lock-free per-pool
    /// atomics (`O(registered formats)`, ~58 entries most empty) — cheap enough
    /// for the per-wave `kv-pool` diagnostic. See [`GpuArenaFormatStats`].
    pub(crate) fn gpu_format_stats(&self) -> GpuArenaFormatStats {
        let mut s = GpuArenaFormatStats::default();
        for (key, pool) in self.inner.pools.iter() {
            if key.location != ArenaLocation::Gpu {
                continue;
            }
            let arenas = pool.total_arenas.load(Ordering::Relaxed);
            if arenas == 0 {
                continue;
            }
            let reserved = arenas.saturating_mul(TARGET_ARENA_BYTES);
            // Slab is ~TARGET_ARENA_BYTES regardless of format, so per-chunk
            // bytes = slab / chunks-per-slab; live bytes = occupied slots × that.
            let per_chunk = TARGET_ARENA_BYTES
                .checked_div(pool.arena_chunks)
                .unwrap_or(0);
            let live = pool.total_live().saturating_mul(per_chunk);
            if matches!(key.format, KvFormat::Float(_)) {
                s.float_arenas += arenas;
                s.float_reserved_bytes += reserved;
                s.float_live_bytes += live;
            } else {
                s.quant_arenas += arenas;
                s.quant_reserved_bytes += reserved;
                s.quant_live_bytes += live;
            }
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
    fn test_gpu_format_stats_splits_float_and_quant() {
        use crate::kv_cache::QuantFormat;
        let pool = ChunkGidPool::new();
        let fkey = ArenaKey::gpu_float(DType::BF16);
        let qkey = ArenaKey::gpu_quant(QuantFormat::Q8_0);

        // Two float arenas with three live slots; one quant arena with five.
        pool.register_arena(fkey.clone());
        pool.register_arena(fkey.clone());
        pool.register_arena(qkey.clone());
        let _f: Vec<_> = (0..3)
            .map(|_| pool.allocate_for(fkey.clone()).unwrap())
            .collect();
        let _q: Vec<_> = (0..5)
            .map(|_| pool.allocate_for(qkey.clone()).unwrap())
            .collect();

        let s = pool.gpu_format_stats();

        // Per-chunk bytes derive from the same slab-capacity helper the accessor
        // uses, so the expected live bytes are exact, not tolerance-based.
        let f_per_chunk =
            TARGET_ARENA_BYTES / arena_chunks_for_format(KvFormat::Float(DType::BF16));
        let q_per_chunk =
            TARGET_ARENA_BYTES / arena_chunks_for_format(KvFormat::Quantized(QuantFormat::Q8_0));

        assert_eq!(s.float_arenas, 2);
        assert_eq!(s.float_reserved_bytes, 2 * TARGET_ARENA_BYTES);
        assert_eq!(s.float_live_bytes, 3 * f_per_chunk);
        assert_eq!(s.quant_arenas, 1);
        assert_eq!(s.quant_reserved_bytes, TARGET_ARENA_BYTES);
        assert_eq!(s.quant_live_bytes, 5 * q_per_chunk);
    }

    #[test]
    fn test_gpu_format_stats_empty_pool_is_zero() {
        // A pool with registered-but-unallocated formats reports nothing: the
        // per-wave diagnostic must not count preallocated empty pool-table
        // entries as resident arenas.
        let pool = ChunkGidPool::new();
        let s = pool.gpu_format_stats();
        assert_eq!(s.float_arenas, 0);
        assert_eq!(s.quant_arenas, 0);
        assert_eq!(s.float_reserved_bytes, 0);
        assert_eq!(s.quant_reserved_bytes, 0);
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

        // Allocating again should reuse it — the freed slot is popped straight
        // back off the recycle stack, so the same gid comes out.
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
        pool.register_arena(key.clone());

        let n_threads = 12;
        let batch = 400; // batches of live gids so arenas fill and empty
        let rounds = 40;
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let pool = Arc::clone(&pool);
                let key = key.clone();
                thread::spawn(move || {
                    for _ in 0..rounds {
                        let mut held = Vec::with_capacity(batch);
                        for _ in 0..batch {
                            // Mirror `alloc_chunk_for_key`: register on exhaustion.
                            let g = loop {
                                if let Some(g) = pool.allocate_for(key.clone()) {
                                    break g;
                                }
                                pool.register_arena(key.clone());
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
        let cap = pool.free_list_len_for(key.clone());
        assert!(cap > 0);
        let mut drained = Vec::new();
        while let Some(g) = pool.allocate_for(key.clone()) {
            drained.push(g);
        }
        assert_eq!(
            drained.len(),
            cap,
            "bitmap + fallback must reach every free slot after concurrent churn"
        );
    }
}
