//! Holds the state shared by every layer of one backing group.
//!
//! Storage, the chunk allocator, the record arenas, the shared identity palette
//! and unit scales, and the per-layer registry all live here, so arenas are pooled
//! across layers rather than duplicated. A mid-wave refusal that needs an arena
//! records its size class, so the arena is created in a gap between forwards
//! instead of inside the allocator. Empty arenas are released in two phases,
//! storage first and then the index, so a reissued index cannot be adopted while
//! its old arena is still standing.

use super::*;

pub(super) fn register_state(inner: &Arc<BackingInner>, state: &Arc<RwLock<BlockTableState>>) {
    if let Ok(mut registry) = inner.state_registry.write() {
        registry.retain(|_, w| w.strong_count() > 0);
        registry.insert(state.read().unwrap().layer_idx, Arc::downgrade(state));
    }
}

/// Inner state of ChunkedKvBacking, wrapped in Arc for registry tracking.
///
/// Shared across all layers that have the same (n_kv_head, head_dim, format)
/// so that arenas are pooled globally rather than duplicated per-layer.
#[derive(Debug)]
pub(crate) struct BackingInner {
    pub(crate) storage: ArenaStorage,
    /// All per-layer sequence states sharing this storage. Needed so a physical
    /// GID move can patch every live host-side reference atomically.
    pub(super) state_registry: RwLock<ahash::HashMap<usize, Weak<RwLock<BlockTableState>>>>,
    /// Shared chunk GID allocation pool across all layers.
    pub(crate) pool: super::super::gid_pool::ChunkGidPool,
    pub(crate) device: Device,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    /// Pre-built identity palette buffer shared by all fresh ChunkWindows.
    /// Size: `n_kv_head * (head_dim / 4).max(1)` bytes. Each head's slice is
    /// identical — the standard (0,0,1,1,2,2,3,3,...) 2-bit palette mapping.
    pub(crate) identity_pal: Arc<Vec<u8>>,
    /// Pre-built unity outer-scale buffer shared by all fresh ChunkWindows
    /// (encoder multiplies by 1.0 = no-op, decoder divides by 1.0 = no-op).
    /// Size: `n_kv_head * N_PALETTE` f32 values, every entry 1.0. Cloning the
    /// `Arc` is cheap — no per-chunk heap allocation.
    pub(crate) identity_scale: Arc<Vec<f32>>,
    /// Per-`(head, palette)` K-side format tag for a **freshly allocated writer
    /// chunk** — the active format, not the configured sealed one. On GPU that
    /// is `R16`; a chunk does not reach its configured format until its turn
    /// seals and quantizes (see [`active_kv_formats`]). Shared `Arc` so
    /// `alloc_block_chunks` pays no per-chunk allocation, exactly like
    /// `identity_pal`.
    ///
    /// [`active_kv_formats`]: crate::kv_cache::active_kv_formats
    pub(crate) active_k_fmt: Arc<Vec<u8>>,
    /// V-side counterpart of [`Self::active_k_fmt`] (`F16` on GPU).
    pub(crate) active_v_fmt: Arc<Vec<u8>>,
    /// Reusable pinned-memory stager for async H2D metadata uploads.
    pub(crate) pinned_stager: PinnedStager,
    /// Device-resident per-chunk KV-head metadata records. Built at quantize,
    /// cold-load, and warm→hot elevate; read by the attention kernels via each
    /// slice's `kvheads_ptr`. Shared across all layers of this group.
    pub(crate) meta_pool: super::super::meta_pool::MetaPool,

    /// Single-latent (K≡V) mode: one latent vector serves as both key and
    /// value (DeepSeek MLA-style attention). Chunk allocation aliases each V
    /// band GID to its K band GID (the refcounted `ChunkGid` handles make the
    /// double reference safe), and contiguous writes skip the V plane — so V
    /// storage costs nothing and every table/kernel consumer sees
    /// `v_ptr == k_ptr` without further special-casing. Set once, before any
    /// chunk is allocated.
    pub(crate) single_latent: std::sync::atomic::AtomicBool,
    /// Sticky: some sealed chunk carries a non-identity pal_map (the latent
    /// band compressor regrouped it) — attention launches must use the
    /// MAPPED kernel instantiation.
    pub(crate) mapped_sealed: std::sync::atomic::AtomicBool,

    /// Size classes a mid-wave refusal wanted an arena for.
    ///
    /// **The refusal's only useful output.** A sealing pass discovers it needs a
    /// new arena at the leaf of the allocator, several frames deep and with
    /// locks held, and is told to come back later — but "come back later" on its
    /// own means rediscovering the same need at the same depth on the next pass,
    /// after redoing the selection work that led there. Recording the class
    /// turns the refusal into an instruction: create *this*, in the gap, and the
    /// next pass will find it and only ever fill it, which is allowed at any
    /// time (`BackingInner::create_deferred_arenas`).
    ///
    /// A `Vec` rather than a set because `ArenaKey` is two small enums and the
    /// ladder has eight classes — membership is a linear scan over single digits.
    // Drained by `create_deferred_arenas`, which only has work to do when there
    // are device arenas to create.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) deferred_arenas: Mutex<Vec<super::super::arena::ArenaKey>>,
    /// Arenas `create_deferred_arenas` made ahead of a demand, with the gaps
    /// each has waited: protected from the empty sweep until the demand fills
    /// them or is abandoned (`BackingInner::settle_fresh`). Unprotected, the
    /// sweep that runs in the same gap took them back before the pass they
    /// were made for could fill them.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fresh_arenas: Mutex<Vec<(super::super::arena::ArenaKey, usize, u32)>>,
}

impl BackingInner {
    /// Per-head band count for the arena/record layout: [`LATENT_N_BANDS`] on
    /// the single-latent path, [`N_PALETTE`] for GQA. Every band-count use in
    /// allocation, arena sub-band sizing, and record serialization goes through
    /// this so the single-latent path can carry [`LATENT_N_BANDS`] bands while GQA
    /// stays at [`N_PALETTE`].
    pub(crate) fn n_palette(&self) -> usize {
        if self
            .single_latent
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            crate::kv_cache::arena_table::LATENT_N_BANDS
        } else {
            crate::kv_cache::arena_table::N_PALETTE
        }
    }

    #[allow(dead_code)]
    pub(super) fn registered_states(&self) -> Vec<Arc<RwLock<BlockTableState>>> {
        if let Ok(mut registry) = self.state_registry.write() {
            registry.retain(|_, w| w.strong_count() > 0);
            registry.values().filter_map(|w| w.upgrade()).collect()
        } else {
            Vec::new()
        }
    }

    /// Get a reference to the state for a given layer, if it exists.
    pub(crate) fn layer_ref(&self, layer_idx: usize) -> Option<Arc<RwLock<BlockTableState>>> {
        if let Ok(registry) = self.state_registry.read() {
            if let Some(weak) = registry.get(&layer_idx) {
                if let Some(state) = weak.upgrade() {
                    return Some(state);
                }
            }
        }
        None
    }

    #[allow(dead_code)]
    pub(crate) fn layer_tree(&self) -> ahash::HashMap<usize, Arc<RwLock<BlockTableState>>> {
        if let Ok(registry) = self.state_registry.read() {
            registry
                .iter()
                .filter_map(|(&idx, w)| w.upgrade().map(|s| (idx, s)))
                .collect()
        } else {
            ahash::HashMap::new()
        }
    }

    /// The lowest hole below the KV frontier that `key`'s arenas could move
    /// down into: the lowest free region of a device arena's reservation, read
    /// afresh on each call. `None` for a host arena, which stands on no region.
    pub(in super::super) fn lowest_hole(&self, key: ArenaKey) -> Option<usize> {
        #[cfg(feature = "cuda")]
        if key.location == ArenaLocation::Gpu {
            if let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() {
                return super::super::region_pool::lowest_free_region(gpu_id);
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = key;
        None
    }

    pub(in super::super) fn release_empty_arenas(&self) -> Result<usize> {
        let mut freed = 0;

        // Phase 1: Pool-driven tombstoning of fully-free arenas.
        // For each format key, repeatedly call next_tombstone until exhausted.
        let keys = self.pool.format_keys();
        for key in keys {
            while let Some(arena_idx) = self.pool.next_tombstone(key, self.lowest_hole(key)) {
                // Storage first, then the index: reissued any earlier, a registration
                // could adopt the arena still standing at it (see `next_tombstone`).
                // A release that fails leaves the index out of circulation for good —
                // storage may still hold the arena, so reusing it is the bug itself.
                self.storage.release_arena(arena_idx)?;
                self.pool.recycle_arena_index(arena_idx);
                // Paired with the recycle log in `ChunkGidPool::register_arena`:
                // a fault correlated between a free here and a re-registration
                // of the same index is the index-re-tenancy signature.
                tracing::debug!(
                    target: "candle_nn::kv_cache::gid_pool",
                    arena_idx,
                    ?key,
                    "arena index freed (empty sweep)"
                );
                freed += 1;
            }
        }

        // Phase 2: Truncate unused tail arenas (tombstoned or otherwise).
        // Use the pool's registry to find the highest live arena.
        let max_used_chunk = self.pool.max_gid().unwrap_or(-1);

        let needed_arenas = if max_used_chunk < 0 {
            0
        } else {
            ((max_used_chunk as usize) / GID_STRIDE) + 1
        };

        let current_arenas = self.storage.arena_count()?;
        if needed_arenas < current_arenas {
            let tail_freed = current_arenas - needed_arenas;
            self.storage.truncate_arenas(needed_arenas)?;
            // Keep pool's recycle queue in sync: indices that were in free_arenas
            // but are now beyond the new storage boundary must not be handed out
            // again (a later registration growing storage would fill them with the
            // wrong format, and ensure_arena_exists would skip creation because
            // current >= needed).
            self.pool.drain_free_arenas_above(needed_arenas);

            freed += tail_freed;
        }

        self.pool.resync_counters();
        // A sweep that frees nothing while the pool still reports recoverable
        // arenas is the wedge signature: relief keeps being asked for memory
        // the pool believes it has and cannot hand over. Say so rather than
        // leaving `arenas_released=0` unexplained.
        if freed == 0 && self.pool.can_reclaim_arena() {
            tracing::debug!(
                target: "candle_nn::kv_cache::gid_pool",
                "empty sweep freed nothing though the pool reports a recoverable arena"
            );
        }
        Ok(freed)
    }

    /// Release one arena the caller created, if nothing is in it; `true` when it did.
    ///
    /// The targeted form of [`Self::release_empty_arenas`], for a creator handing back
    /// what it made and did not use without also releasing everything else that happens
    /// to be empty. The pool entry and the storage go together, as in the sweep.
    pub(in super::super) fn release_arena_if_empty(
        &self,
        key: ArenaKey,
        arena_idx: usize,
    ) -> Result<bool> {
        if !self.pool.tombstone_if_empty(key, arena_idx) {
            return Ok(false);
        }
        self.storage.release_arena(arena_idx)?;
        self.pool.recycle_arena_index(arena_idx);
        self.pool.resync_counters();
        Ok(true)
    }
}

impl BackingInner {
    /// Update a single block's per-head GID vector AND its per-block palette
    /// maps and outer scales, then refresh the GPU slot buffer so the decode
    /// kernel sees the updated state.
    ///
    /// `k_pal` / `v_pal` must be `n_kv_head × (head_dim / 4)` bytes each
    /// (or empty for identity routing).
    /// `k_scale` / `v_scale` must each contain `n_kv_head × N_PALETTE` f32
    /// values (or empty for unity = no outer scaling).
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn set_block_gids_sharded_and_update_gpu(
        &self,
        layer_idx: usize,
        batch_idx: usize,
        block_idx: usize,
        gids: HeadGids,
        k_pal: std::sync::Arc<Vec<u8>>,
        v_pal: std::sync::Arc<Vec<u8>>,
        k_scale: std::sync::Arc<Vec<f32>>,
        v_scale: std::sync::Arc<Vec<f32>>,
        k_fmt: std::sync::Arc<Vec<u8>>,
        v_fmt: std::sync::Arc<Vec<u8>>,
        arena_info: &[ResolvedArenaInfo],
    ) -> Result<()> {
        let state = self.layer_ref(layer_idx).ok_or_else(|| {
            candle::Error::Msg(format!(
                "layer_idx {} not found in backing state registry",
                layer_idx
            ))
        })?;
        let mut state = state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        if let Some(slot) = state.sequences.get_mut(batch_idx).and_then(|s| s.as_mut()) {
            slot.set_block_gids(
                block_idx, gids, k_pal, v_pal, k_scale, v_scale, k_fmt, v_fmt,
            );
            slot.update_gpu_chunk(block_idx, self.n_kv_head, self.head_dim, arena_info)?;
        }
        Ok(())
    }
}
