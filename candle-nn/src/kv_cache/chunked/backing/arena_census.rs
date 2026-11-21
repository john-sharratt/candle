//! Reads the arena census a compaction plans from, and the queries around it.
//!
//! The census joins a pool's occupancy, from the refcount tables, to the physical
//! position of each arena, since occupancy alone cannot say which way is down.
//! Arenas with no region, which are host arenas, are skipped. Cheap ranking
//! queries answer which pool holds the frontier without spending the census
//! budget, and `release_empty_arenas` hands back fully empty arenas without
//! moving any chunk.

use super::*;

impl ChunkedKvBacking {
    /// Release any fully-empty arenas back to the pool **without** the
    /// expensive chunk-moving defrag — the cheap half of compaction. Returns
    /// arenas freed. Used by the scheduler's pressure path so it never spends
    /// seconds on a speculative defrag (which, during active prefills, is
    /// usually futile — the free space sits in protected arenas); the costly
    /// defrag stays on the reactive allocation-time OOM retry.
    pub fn release_empty_arenas(&self) -> Result<usize> {
        self.inner.release_empty_arenas()
    }
}

/// One band a live slot names, as the refcount tables are keyed.
///
/// Carries the `(arena, slot)` pair rather than an address because that is the form
/// the occupancy tables answer in, and the address is derivable from it while the
/// reverse needs the arena extents. `record` is the chunk's coresident record, so a
/// caller can check what *it* names as well as what the gid names — the two are
/// independent recordings of one location and either can be the stale one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveBand {
    pub arena: usize,
    pub slot_in_arena: usize,
    pub record: Option<u64>,
    /// Position in the chunk's `HeadGids` grid, which is the index the record's
    /// pointer words are laid out by — `h * n_palette * 2 + p * 2 + is_value`.
    ///
    /// Carried because holes are skipped: a caller counting the bands it was handed
    /// would attribute each one to the wrong record word as soon as any gid in the
    /// grid is absent, and then report every band after the hole as disagreeing.
    pub band: usize,
}

impl ChunkedKvBacking {
    pub fn resolve_arena_info(&self) -> Result<Vec<ResolvedArenaInfo>> {
        self.inner.resolve_arena_info()
    }

    /// Like [`Self::resolve_arena_info`] but only resolves the device pointers
    /// for the arenas in `needed`, leaving the rest at the zero default. The
    /// return is still a dense `arena_idx`-indexed Vec, so callers index it the
    /// same way — they just must only read the arenas they asked for. The
    /// per-arena `to_arena_entry` pointer resolve is the pass's dominant `other`
    /// cost at pressure (O(num_arenas)); a reassemble only touches the handful of
    /// dst arenas it just allocated, so this collapses it to O(touched).
    pub fn resolve_arena_info_for(
        &self,
        needed: &std::collections::HashSet<usize>,
    ) -> Result<Vec<ResolvedArenaInfo>> {
        self.inner.resolve_arena_info_for(needed)
    }

    /// Get the number of arenas in backing storage.
    pub fn arena_count(&self) -> Result<usize> {
        self.inner.storage.arena_count()
    }

    /// Occupied slots per arena for one pool, from the refcount tables.
    ///
    /// The authority on whether a slot is free, exposed because the integrity checks
    /// must not reconstruct that from holders — every attempt to do so has missed one
    /// (arenas pool globally across same-config layers, and sealed sequences hold
    /// slots too), and an incomplete reference set reports faults that are not there.
    pub fn pool_occupancy(&self, key: ArenaKey) -> Vec<(usize, usize, Vec<u32>)> {
        self.inner.pool.pool_occupancy(key)
    }

    /// Whether any region below the arena frontier is free — so whether a fresh arena
    /// would land *under* the frontier rather than raise it.
    ///
    /// The region free list is lowest-index-first, so a claim takes the lowest hole when
    /// one exists. Gates every arena a compaction creates ahead of demand — the per-pool
    /// low arenas and the record reservation.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn has_region_hole(&self) -> bool {
        let candle::DeviceLocation::Cuda { gpu_id } = self.device().location() else {
            return false;
        };
        super::super::region_pool::region_stats(gpu_id).is_some_and(|s| s.live_watermark > s.live)
    }

    /// Without a device there is no region pool and so no hole to find.
    #[cfg(not(feature = "cuda"))]
    pub(in super::super) fn has_region_hole(&self) -> bool {
        false
    }

    /// Of the regions `held` names, those holding the frontier that a fresh low arena
    /// could empty, highest first — see `region_pool::frontier_relocations`.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn regions_to_relocate(&self, held: &HashSet<usize>) -> Vec<usize> {
        let candle::DeviceLocation::Cuda { gpu_id } = self.device().location() else {
            return Vec::new();
        };
        super::super::region_pool::regions_to_relocate(gpu_id, held).unwrap_or_default()
    }

    /// Regions held by this device's **record** arenas.
    ///
    /// In-use ground that no size-class row accounts for: `gpu_class_stats` reports only
    /// `ArenaKind::Band` pools, deliberately, so a record arena pollutes neither the
    /// fragmentation figure nor the relief decisions that read it. That leaves it invisible
    /// to a consumer summing the class rows — which then charges it as waste. Measured on
    /// the 30B: 11 regions of 62 "lost" were record arenas holding live records.
    ///
    /// Counted from the pool's registrations rather than from live slots, because an arena
    /// holds its region until it is *released*, not until its last record goes.
    pub fn record_arena_regions(&self) -> usize {
        super::super::arena::RECORD_STRIDES
            .iter()
            .map(|&stride| {
                let key = ArenaKey::for_record_stride(ArenaLocation::Gpu, stride);
                self.inner.pool.pool_arena_load(key).len()
            })
            .sum()
    }

    /// `(arena index, capacity, live slots)` per arena of one pool.
    ///
    /// The cheap form of [`Self::pool_occupancy`] — a count instead of the list — for
    /// a caller that only needs to know which arenas a pool has registered. The
    /// compaction's record walk asks it of every record-stride pool, where the list of
    /// occupied slots would be a per-slot upload the walk exists to avoid.
    pub fn pool_arena_load(&self, key: ArenaKey) -> Vec<(usize, usize, usize)> {
        self.inner.pool.pool_arena_load(key)
    }

    /// Every band every live slot names, for the integrity checks.
    ///
    /// Host-only and complete: it walks the block tables rather than sampling them, so
    /// a gid naming ground nothing holds cannot hide in the part that was not looked
    /// at. Returns the band's `(arena, slot)` — which is what the refcount tables are
    /// keyed by — together with the chunk's record address, so a caller that wants to
    /// check the record's pointer as well knows where to read it.
    ///
    /// `record` is `None` for a chunk with no coresident record: a freshly-allocated
    /// float writer window, or one whose record was dropped when its gids were
    /// replaced. Both are ordinary and neither needs checking.
    pub fn live_bands(&self) -> Result<Vec<(usize, Vec<LiveBand>)>> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let mut out = Vec::new();
        for (slot, entry) in state.sequences.iter().enumerate() {
            let Some(seq) = entry.as_ref() else { continue };
            let mut bands = Vec::new();
            for cw in seq.chunks_slice().iter() {
                let record = cw
                    .meta
                    .as_ref()
                    .map(|m| m.device_addr())
                    .filter(|a| *a != 0);
                for (band, g) in cw.gids.as_slice().iter().enumerate() {
                    if g.raw() < 0 {
                        continue;
                    }
                    bands.push(LiveBand {
                        arena: g.arena_idx(),
                        slot_in_arena: g.chunk_idx(),
                        record,
                        band,
                    });
                }
            }
            if !bands.is_empty() {
                out.push((slot, bands));
            }
        }
        Ok(out)
    }

    /// Each GPU arena's base address, slot stride and capacity.
    ///
    /// What a caller needs to turn a raw device address back into the `(arena, slot)`
    /// the refcount tables are keyed by — the inverse of `slot_addr`, which is the
    /// direction an orphaned *record pointer* has to be checked in.
    pub fn arena_extents(&self) -> Result<Vec<(usize, u64, i64, u32)>> {
        let mut needed = std::collections::HashSet::new();
        for class in SizeClass::all() {
            let key = ArenaKey::new(class, ArenaLocation::Gpu);
            for (arena, _, _) in self.inner.pool.pool_arena_load(key) {
                needed.insert(arena);
            }
        }
        let info = self.resolve_arena_info_for(&needed)?;
        Ok(needed
            .into_iter()
            .filter_map(|a| {
                let r = info.get(a)?;
                (r.base_ptr != 0 && r.chunk_byte_stride > 0).then_some((
                    a,
                    r.base_ptr,
                    r.chunk_byte_stride,
                    r.chunk_capacity,
                ))
            })
            .collect())
    }

    /// `(arenas_held, arenas_if_packed)` for one pool, from the refcount tables' live
    /// counters — see [`ChunkGidPool::pool_sparsity`].
    pub fn pool_sparsity(&self, key: ArenaKey) -> (usize, usize) {
        self.inner.pool.pool_sparsity(key)
    }

    /// The highest region rank a live arena of `key` holds, and how many slots that
    /// arena is holding — the cheap question "is this pool what is holding the
    /// frontier up?".
    ///
    /// **Cheap on purpose.** A compaction pass is time-budgeted, and its budget is
    /// spent on the occupancy census; a pass that censuses the ladder in class order
    /// spends that budget on whichever rungs happen to come first and can run out
    /// before reaching the pool that owns the topmost arena — which is the only pool
    /// whose position costs the weight side anything. So the ordering decision has to
    /// be answerable without the census, which is what
    /// [`ChunkGidPool::pool_arena_load`] is for.
    pub fn pool_top_rank(&self, key: ArenaKey) -> Option<(usize, usize)> {
        let load = self.inner.pool.pool_arena_load(key);
        if load.is_empty() {
            return None;
        }
        self.inner
            .storage
            .read(|s| {
                let arenas = s.arenas();
                load.iter()
                    .filter(|(_, _, live)| *live > 0)
                    .filter_map(|(idx, _, live)| {
                        arenas
                            .get(idx)
                            .and_then(|a| a.region_rank())
                            .map(|r| (r, *live))
                    })
                    .max_by_key(|(rank, _)| *rank)
            })
            // A poisoned storage lock is not a ranking question; the pass falls back
            // to ladder order, which is where it started.
            .ok()
            .flatten()
    }

    /// The highest region any KV arena on this device holds — every band pool and
    /// the record pool together — or `None` with no KV arena.
    ///
    /// The question a compaction gate asks before it pays for a census: a KV pass
    /// can lower the frontier only when a KV arena is what stands at it. When the
    /// topmost live region is another tenant's, packing the KV pools moves holes
    /// about beneath it and the frontier stays where it is. Cheap for the reason
    /// [`Self::pool_top_rank`] is.
    ///
    /// **An arena emptied and not yet swept still counts.** It holds its region
    /// until a sweep returns it, so while it stands at the frontier the frontier is
    /// the KV side's to lower — the KV pass's sweep is what lowers it. Counted only
    /// by its live slots, it read as another tenant's region, sent the gate to a
    /// span-tenant pass that could not move it, and stalled the gate on that.
    pub fn kv_top_rank(&self) -> Option<usize> {
        let record_key = if self.records_are_resident() {
            self.record_layout().ok().map(|l| l.key)
        } else {
            None
        };
        SizeClass::all()
            .map(|class| ArenaKey::new(class, ArenaLocation::Gpu))
            .chain(record_key)
            .filter_map(|key| self.pool_top_region(key))
            .max()
    }

    /// The highest region rank any arena of `key` holds, live slots or none.
    pub(super) fn pool_top_region(&self, key: ArenaKey) -> Option<usize> {
        let load = self.inner.pool.pool_arena_load(key);
        if load.is_empty() {
            return None;
        }
        self.inner
            .storage
            .read(|s| {
                let arenas = s.arenas();
                load.iter()
                    .filter_map(|(idx, _, _)| arenas.get(idx).and_then(|a| a.region_rank()))
                    .max()
            })
            .ok()
            .flatten()
    }

    /// The census one pool's compaction plans from — occupancy joined to physical
    /// address order.
    ///
    /// Two halves, from two owners: the gid pool knows which slots are occupied
    /// ([`ChunkGidPool::pool_occupancy`]) and the arena storage knows where each
    /// arena physically sits ([`Arena::region_rank`]). Only the join is
    /// plannable — occupancy alone cannot say which way is "down".
    ///
    /// **An arena with no region is skipped.** That is every CPU arena, whose
    /// slab is an ordinary host allocation with no position in the reservation
    /// and so nothing to pack toward. A pass over a host pool is therefore a
    /// no-op rather than a plan over meaningless ranks, and the empty result says
    /// so. (`plan_pool` filters by `ArenaKey`, so a host pool is never mixed into
    /// a device one; this is the second, independent guard.)
    ///
    /// Taken under one `storage.read`, but the occupancy bitmaps are sampled
    /// word by word — see [`ArenaRefcounts::occupied_slots`] for why a torn read
    /// is tolerable and what finally refuses a slot the census got wrong.
    pub fn compaction_census(&self, key: ArenaKey) -> Result<Vec<ArenaSlots>> {
        let occupancy = self.inner.pool.pool_occupancy(key);
        if occupancy.is_empty() {
            return Ok(Vec::new());
        }
        self.inner.storage.read(|s| {
            let arenas = s.arenas();
            let mut out = Vec::with_capacity(occupancy.len());
            for (arena_idx, capacity, occupied) in &occupancy {
                let Some(arena) = arenas.get(arena_idx) else {
                    // Registered in the pool but gone from storage: a release
                    // that has not finished propagating. Nothing to move.
                    continue;
                };
                let Some(rank) = arena.region_rank() else {
                    continue;
                };
                out.push(ArenaSlots {
                    arena_idx: *arena_idx,
                    key,
                    rank,
                    capacity: *capacity,
                    occupied: occupied.clone(),
                });
            }
            out
        })
    }

    /// The census entry for an arena this pass has just created and nothing occupies:
    /// its physical rank and capacity, with no occupied slots.
    ///
    /// What [`Self::compaction_census`] would report for it, without re-walking the rest
    /// of the pool's bitmaps to find out. `None` when the arena has no region (a host
    /// arena) or is not in storage.
    #[allow(dead_code)] // the only caller is `compact`, which is cuda-gated
    pub(in super::super) fn fresh_arena_slots(
        &self,
        key: ArenaKey,
        arena_idx: usize,
    ) -> Option<ArenaSlots> {
        let rank = self
            .inner
            .storage
            .read(|s| s.arenas().get(&arena_idx).and_then(|a| a.region_rank()))
            .ok()
            .flatten()?;
        Some(ArenaSlots {
            arena_idx,
            key,
            rank,
            capacity: key.chunks(),
            occupied: Vec::new(),
        })
    }

    /// Execute a read operation on the arena storage.
    /// This is the preferred API for kernels that need to handle heterogeneous storage.
    pub fn with_arenas<R>(&self, f: impl FnOnce(&ahash::AHashMap<usize, Arena>) -> R) -> Result<R> {
        self.inner.storage.read(|s| f(s.arenas()))
    }

    /// Execute a write operation on the arena storage.
    #[allow(dead_code)] // Used for in-place arena mutations (quantization, offload)
    pub(crate) fn with_arenas_mut<R>(
        &self,
        f: impl FnOnce(&mut ArenaStorageState) -> R,
    ) -> Result<R> {
        self.inner.storage.write(f)
    }

    /// Execute a fallible write operation on the arena storage.
    #[allow(dead_code)] // Used for fallible arena mutations (migration, resize)
    pub(crate) fn try_with_arenas_mut<R>(
        &self,
        f: impl FnOnce(&mut ArenaStorageState) -> Result<R>,
    ) -> Result<R> {
        self.inner.storage.try_write(f)
    }

    /// Get the max_blocks for this backing (width of the block table).
    pub fn max_blocks(&self) -> usize {
        self.state
            .read()
            .expect("chunked state lock poisoned")
            .max_blocks
    }

    /// Borrow the backing's device. Used by call sites that need to
    /// hand a `&Device` to one of the kernel-driving helpers without
    /// threading it down from the caller.
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// Collect arena diagnostic rows for this backing.
    pub(crate) fn arena_rows(&self) -> Result<Vec<crate::kv_cache::chunked::arena::ArenaRow>> {
        let mut rows = self.inner.storage.arena_rows()?;
        // Patch in pool-derived counts (storage sets active=0, free_list=0 as placeholders)
        for row in &mut rows {
            let free = self.inner.pool.arena_free_count(row.arena_idx) as usize;
            row.free_list = free;
            row.active = row.capacity.saturating_sub(free);
        }
        Ok(rows)
    }
}
