//! The lock-free refcount table of one arena.
//!
//! A slot's word is its refcount while it is occupied and its free-list link
//! while it is free, and a separate occupancy bit says which. Claims are gated to
//! a single popper, which keeps the recycle stack ABA-free, while drops only push
//! and never wait. A reader that sees a slot free never sees a stale count, and a
//! creation window keeps a tombstoner off an arena whose creator has not yet
//! claimed.

use super::*;

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
    pub(super) counts: Vec<AtomicU16>,
    /// Occupancy: one BIT per slot, `set ⟺ slot is allocated`. With `counts`
    /// overlapped (refcount xor link) this is now the **authoritative** free/
    /// occupied discriminator (not just a scan hint): set on claim, cleared on
    /// the `1→0` last drop, `fetch_or`/`fetch_and` so different bits of a word
    /// compose. Read by [`ArenaRefcounts::occupied_slots`] to enumerate live
    /// slots for a compaction census.
    pub(super) occupancy: Vec<AtomicU64>,
    /// Lock-free intrusive recycle stack of freed slots (Treiber). Links live
    /// in `counts` (a free slot's word = next-free index, or `arena_chunks` =
    /// bottom). Packed head: low 32 bits = top slot index (`arena_chunks` ⇒
    /// empty), high 32 bits = ABA version tag. `dec` pushes a freed slot; alloc
    /// pops — both O(1), no scan, fragmentation-immune. Allocs are gated
    /// (single popper), so ABA can't actually arise; the tag is belt-and-braces.
    pub(super) recycle_head: AtomicU64,
    /// High-water mark: index of the next never-allocated slot. Fresh capacity
    /// comes from bumping this (so a new arena needs no O(A) free-list init —
    /// the "one range `[0, A)`"); only *freed* slots ride `recycle_head`.
    /// Written only by the gated allocator, so plain relaxed load/store.
    pub(super) hwm: AtomicU32,
    /// Number of currently-allocated slots in this arena (slots with
    /// `counts[i] > 0`). Maintained on each successful alloc/drop
    /// transition. Used by `try_tombstone` (arena is tombstoneable
    /// when `live == 0`) and diagnostics.
    pub(super) live: AtomicUsize,
    /// Pool-wide live counter, shared with [`ArenaPool::total_live`]
    /// and every other arena in the same per-format pool. Updated on
    /// the same `0 → 1` / `1 → 0` transitions as `live`, so
    /// `ArenaPool::total_live()` is an O(1) atomic load instead of a
    /// per-call sum across arenas.
    pub(super) pool_total_live: Arc<AtomicUsize>,
    /// Total chunk capacity of this arena (constant for the arena's
    /// lifetime, set at `register_arena` time).
    pub(super) arena_chunks: usize,
    /// Global arena index — encoded into every gid as the upper bits.
    pub(super) arena_idx: usize,
    /// Format key this arena was registered with. Inlined here so
    /// `ChunkGid::route_key` is a single pointer-deref away.
    pub(super) key: ArenaKey,
    /// Shared pool arena-capacity bitmap. `dec` sets this arena's bit on a
    /// full → non-full transition so the pool's `allocate_any` can find it via
    /// find-first-set instead of scanning every arena.
    pub(super) capacity: Arc<CapacityBitmap>,
    /// This arena's place in its pool's placement order — its region rank once
    /// [`ArenaPool::set_rank`] records it, [`unranked_place`] until then — and
    /// so the bit it owns in `capacity`.
    pub(super) place: AtomicUsize,
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
    pub(super) creation_pending: AtomicBool,
}

impl ArenaRefcounts {
    pub(super) fn new(
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
    pub(super) fn place(&self) -> usize {
        self.place.load(Ordering::Acquire)
    }

    /// Whether this arena is still inside its creation window (registered but
    /// no gid allocated yet). `Acquire` pairs with the `Release` clear in
    /// [`Self::occupy`]: a `false` here guarantees the first slot's live
    /// increment is visible.
    #[inline]
    pub(super) fn creation_pending(&self) -> bool {
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
    pub(super) fn end_creation_window(&self) {
        self.creation_pending.store(false, Ordering::Release);
    }

    /// Unpack the recycle head into `(top_slot_idx, version)`; `top == arena_chunks`
    /// means the stack is empty.
    #[inline]
    pub(super) fn head_parts(head: u64) -> (usize, u64) {
        ((head & 0xFFFF_FFFF) as usize, head >> 32)
    }

    /// Pack a `(top_slot_idx, version)` into a recycle-head word.
    #[inline]
    pub(super) fn head_pack(idx: usize, version: u64) -> u64 {
        ((version & 0xFFFF_FFFF) << 32) | (idx as u64 & 0xFFFF_FFFF)
    }

    /// Whether every slot is claimed (`live == arena_chunks`). The authoritative
    /// full check the pool capacity bit mirrors.
    #[inline]
    pub(super) fn is_full(&self) -> bool {
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
    pub(super) fn set_occupied(&self, i: usize) {
        self.occupancy[i / 64].fetch_or(1u64 << (i % 64), Ordering::Relaxed);
    }

    /// Mark slot `i` free in the occupancy discriminator (clear its bit).
    ///
    /// Atomic `fetch_and` so it composes with a concurrent `set_occupied`.
    /// **Release**: `dec` calls this before overwriting `counts[i]` with the
    /// recycle-stack link, so a reader that Acquire-observes the cleared bit
    /// (slot free) never still sees the stale refcount in the word.
    #[inline]
    pub(super) fn set_free(&self, i: usize) {
        self.occupancy[i / 64].fetch_and(!(1u64 << (i % 64)), Ordering::Release);
    }

    /// Finish claiming slot `i`: overwrite its `counts` word (was a free-list
    /// link) with refcount 1, mark it occupied, and bump the live counters.
    /// Runs under `alloc_gate`, and no `ChunkGid` for `i` exists yet, so no one
    /// races these writes.
    #[inline]
    pub(super) fn occupy(&self, i: usize) {
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
    pub(super) fn try_claim_one(&self) -> Option<usize> {
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
    pub(super) fn try_claim_run(&self, len: usize) -> Option<usize> {
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
    pub(super) fn run_fits(&self, len: usize) -> bool {
        (self.hwm.load(Ordering::Relaxed) as usize) + len <= self.arena_chunks
    }

    /// Increment a slot's refcount — called by `ChunkGid::clone`. The
    /// slot must already be allocated (count ≥ 1); we never need
    /// Acquire semantics because the data is already visible.
    #[inline]
    pub(super) fn inc(&self, chunk_idx: usize) {
        self.counts[chunk_idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement a slot's refcount — called by `ChunkGid::drop`. On the `1→0`
    /// last drop, free the slot: clear its occupancy bit, push it onto the
    /// recycle stack (its `counts` word becomes the next-free link), and
    /// decrement the live counters (per-arena and pool-wide).
    #[inline]
    pub(super) fn dec(&self, chunk_idx: usize) {
        let prev = self.counts[chunk_idx].fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            #[cfg(feature = "tensor-assert")]
            super::super::release_watch::on_release(
                (self.arena_idx * GID_STRIDE + chunk_idx) as i64,
            );
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
    pub(super) fn load(&self, chunk_idx: usize) -> u16 {
        self.counts[chunk_idx].load(Ordering::Relaxed)
    }

    /// Number of currently-allocated slots in this arena.
    #[inline]
    pub(super) fn live_count(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    /// Number of currently-free slots in this arena.
    #[inline]
    pub(super) fn free_count(&self) -> usize {
        self.arena_chunks.saturating_sub(self.live_count())
    }
}
