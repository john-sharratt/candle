//! Allocating chunks for a backing: reuse from an arena's free list, and the
//! creation paths for a class that has run dry.
//!
//! A chunk reused from a free list is zeroed first, so the next tenant never
//! reads the previous one's bytes. The allocating paths run with the storage
//! write lock held and so cannot wait for the gap between forwards; their caller
//! holds the arena window instead. A class that is nearly dry asks the next gap
//! for an arena, so the sealer does not run out mid-wave.

use super::*;

impl ChunkedKvBacking {
    /// Allocate a chunk with pre-acquired arena lock.
    /// This version takes a pre-acquired arenas lock to avoid deadlock when called
    /// from contexts that already hold the arena lock.
    ///
    /// **The caller must already hold an [`BackingInner::arena_window`].** Both
    /// branches below can create an arena, and they do it with the storage write
    /// lock held — so they cannot wait for the inter-forward gap themselves
    /// without sleeping on a lock a forward's admit needs.
    pub(in super::super) fn alloc_chunk_with_arenas(
        &self,
        arena_state: &mut ArenaStorageState,
        key: ArenaKey,
    ) -> Result<ChunkGid> {
        if let Some(gid) = self.inner.pool.allocate_for(key) {
            let arena_idx = gid.arena_idx();
            let arena_was_fresh = !arena_state.has_arena(arena_idx);
            if arena_was_fresh {
                let arena = self.create_arena(key, arena_idx)?;
                arena_state.push_arena(arena, arena_idx);
            } else {
                // Free-list reuse on an existing arena: the chunk's bytes are
                // whatever the prior tenant left. Zero them so the new tenant
                // (and any persist quantize pass that reads past token_count)
                // sees clean storage. Fresh arenas are already zero from
                // `Tensor::zeros` at creation, so the arena_was_fresh branch
                // above skips the work.
                self.zero_recycled_chunk(arena_state, arena_idx, gid.chunk_idx())?;
            }
            return Ok(gid);
        }

        // The class is starved. Stamp a region for it; failing that, sweep
        // fully-empty arenas and retry; failing that, take a free slot from a
        // wider class that already has one. Same gates and same order as
        // `BackingInner::stamp_region_promoting`, open-coded because this path
        // already holds the storage lock and so cannot call
        // `ensure_arena_exists` or `release_empty_arenas` — the sweep runs
        // inline against the `&mut ArenaStorageState` the caller lent us.
        let mut key = key;
        let arena_idx = self.inner.pool.register_arena(key);
        // The arena registered here, when there is one, is what the claim below takes
        // from — by name, so its first `occupy` is what closes its creation window.
        // `allocate_for` could land in another arena that freed a slot meanwhile, and
        // this one would then keep its window open with nothing ever claiming from it.
        let mut made = Some(arena_idx);
        if !arena_state.has_arena(arena_idx) {
            match self.create_arena(key, arena_idx) {
                Ok(arena) => arena_state.push_arena(arena, arena_idx),
                Err(e) => {
                    self.inner.pool.force_release_arena(arena_idx);
                    made = None;
                    // Reactive empty sweep, exactly as `stamp_region_promoting`
                    // does before promoting: seal/requantize churn strands
                    // empty arenas that each pin a region, and the periodic
                    // sweeps run between waves/configs — not inside the churn
                    // that is failing right now.
                    let mut swept = 0usize;
                    if self.inner.pool.has_reclaimable() {
                        for k in self.inner.pool.format_keys() {
                            while let Some(idx) =
                                self.inner.pool.next_tombstone(k, self.inner.lowest_hole(k))
                            {
                                arena_state.release_arena(idx);
                                self.inner.pool.recycle_arena_index(idx);
                                swept += 1;
                            }
                        }
                    }
                    let mut created = false;
                    if swept > 0 {
                        let retry_idx = self.inner.pool.register_arena(key);
                        match self.create_arena(key, retry_idx) {
                            Ok(arena) => {
                                arena_state.push_arena(arena, retry_idx);
                                made = Some(retry_idx);
                                created = true;
                            }
                            Err(_) => self.inner.pool.force_release_arena(retry_idx),
                        }
                    }
                    if !created {
                        let mut class = key.class;
                        let mut placed = None;
                        while let Some(next) = class.promote() {
                            class = next;
                            let wider = ArenaKey::new(class, key.location);
                            if let Some(gid) = self.inner.pool.allocate_for(wider) {
                                let ai = gid.arena_idx();
                                drop(gid);
                                if arena_state.has_arena(ai) {
                                    record_class_promotion(key.class, class);
                                    placed = Some(wider);
                                    break;
                                }
                            }
                        }
                        match placed {
                            Some(w) => key = w,
                            None => return Err(e),
                        }
                    }
                }
            }
        }

        let gid = match made {
            Some(idx) => {
                let gid = self.inner.pool.allocate_from_arena(key, idx);
                // A claim that came back empty did not close the window, and an arena
                // whose window stays open is never released.
                self.inner.pool.finish_creation(key, idx);
                gid
            }
            // Promoted: the probe above found a wider arena with room, not a named one.
            None => self.inner.pool.allocate_for(key),
        }
        .ok_or_else(|| candle::Error::Msg("no slot in the region just claimed".into()))?;
        Ok(gid)
    }

    /// Zero one recycled chunk's bytes. Asynchronous on CUDA — the write is
    /// enqueued on the slab's own stream and the call returns once queued.
    /// Same-stream FIFO ordering guarantees the next reader of this chunk sees
    /// the zeros without an explicit fence.
    pub(super) fn zero_recycled_chunk(
        &self,
        arena_state: &mut ArenaStorageState,
        arena_idx: usize,
        chunk_idx: usize,
    ) -> Result<()> {
        let Some(arena) = arena_state.arenas_mut().get_mut(&arena_idx) else {
            return Ok(());
        };
        arena.zero_chunk_at(chunk_idx)
    }

    /// ArenaKey for active (unfilled) K chunks.
    ///
    /// On CUDA we keep the fast R16 active-K path for the decode/prefill kernels.
    /// On CPU we keep active K chunks in float so partial-token writes and tests
    /// do not require block-aligned quantization on every append.
    pub(in super::super) fn active_k_arena_key(&self) -> Result<ArenaKey> {
        let location = self.inner.storage.default_location();
        let (k, _) = crate::kv_cache::active_kv_formats(
            self.inner.storage.k_format(),
            matches!(location, ArenaLocation::Gpu),
        );
        self.inner.arena_key_for(k, location)
    }

    /// ArenaKey for active (unfilled) V chunks — always float.
    pub(in super::super) fn active_v_arena_key(&self) -> Result<ArenaKey> {
        let location = self.inner.storage.default_location();
        let (_, v) = crate::kv_cache::active_kv_formats(
            self.inner.storage.k_format(),
            matches!(location, ArenaLocation::Gpu),
        );
        self.inner.arena_key_for(v, location)
    }

    /// Allocate a full block's worth of flat chunks for the palette4 arenas.
    ///
    /// Returns a `ChunkWindow` with `GIDS_PER_HEAD * n_kv_head` GIDs (N_PALETTE per
    /// head × {K, V}).  Each chunk stores exactly `CHUNK_SIZE * (head_dim / N_PALETTE)`
    /// elements — one head, one palette sub-band, one side.
    ///
    /// HeadGids layout: `head * GIDS_PER_HEAD + palette * 2 + is_value`.
    pub(in super::super) fn alloc_block_chunks(
        &self,
        usage: u32,
        offset: u16,
    ) -> Result<ChunkWindow> {
        // Band count per head: LATENT_N_BANDS (single-latent) or N_PALETTE (GQA).
        let np = self.inner.n_palette();
        let n = np * 2 * self.inner.n_kv_head;
        let k_key = self.active_k_arena_key()?;
        let v_key = self.active_v_arena_key()?;
        // Rope-region key for the single latent: the 64 RoPE dims (bands
        // [LATENT_NOPE_BANDS, np)) are pinned BF16 regardless of the writer
        // format, matching the reference (`nope FP8 ‖ rope BF16`). When the
        // writer format is already BF16 (the wave window) this equals `k_key`,
        // so the store is uniform BF16 — the pre-existing behaviour, only the
        // arena width narrows to the single-latent band width.
        let rope_key = self.inner.arena_key_for(
            KvFormat::Float(DType::BF16),
            self.inner.storage.default_location(),
        )?;
        let mut gids = Vec::with_capacity(n);
        // Per head: one CONTIGUOUS run of N_PALETTE K slots and one of V slots
        // (see `alloc_chunk_run_for_key`). Correctness does NOT depend on the
        // run layout — every kernel addresses each band through its own gid
        // (`resolve_band_source`, and the per-palette KvHead record pointers) —
        // but contiguous bands give the select/QREL walk better spatial
        // locality, so we mint them as runs where a run is available.
        // HeadGids layout stays `head * GIDS_PER_HEAD + palette * 2 + is_value`.
        let single_latent = self
            .inner
            .single_latent
            .load(std::sync::atomic::Ordering::Relaxed);
        if single_latent {
            for _h in 0..self.inner.n_kv_head {
                // Two-region window: bands [0, LATENT_NOPE_BANDS) back the 448-d
                // nope span in the writer format (FP8 E4M3 for the reference
                // config); bands [LATENT_NOPE_BANDS, np) back the 64-d rope tail
                // in BF16. Each region is one contiguous chunk run; the KvHead
                // record still fills all 16 band slots (bands resolve their
                // per-band {ptr, fmt, scale} from their own gid's arena, so the
                // format tag follows the region automatically).
                //
                // K≡V: the V band aliases the K band. `ChunkGid` is a refcounted
                // handle, so the double reference keeps the chunk alive until
                // both drop — V storage costs nothing and every table consumer
                // sees v_ptr == k_ptr.
                let nope_bands = crate::kv_cache::arena_table::LATENT_NOPE_BANDS.min(np);
                let rope_bands = np - nope_bands;
                let nope_run = self.inner.alloc_chunk_run_for_key(k_key, nope_bands)?;
                let rope_run = if rope_bands > 0 {
                    self.inner.alloc_chunk_run_for_key(rope_key, rope_bands)?
                } else {
                    Vec::new()
                };
                for k_gid in nope_run.into_iter().chain(rope_run) {
                    let v_gid = k_gid.clone();
                    gids.push(k_gid);
                    gids.push(v_gid);
                }
            }
        } else {
            // Every head's K run then its V run, claimed as one ordered batch:
            // the same slots in the same order as one claim per run, with the
            // pool's arena checks made once for the chunk rather than 2·n_kv_head
            // times — at 32 heads that bookkeeping was most of a chunk's claim.
            let requests: Vec<(ArenaKey, usize)> = (0..self.inner.n_kv_head)
                .flat_map(|_| [(k_key, np), (v_key, np)])
                .collect();
            let mut runs = self.inner.alloc_chunk_runs_for_keys(&requests)?.into_iter();
            while let (Some(k_run), Some(v_run)) = (runs.next(), runs.next()) {
                for (k_gid, v_gid) in k_run.into_iter().zip(v_run) {
                    gids.push(k_gid);
                    gids.push(v_gid);
                }
            }
        }

        // Active formats — this chunk is a writer and will not reach its
        // configured sealed format until its turn seals and quantizes. GQA
        // shares the constructor's per-(head, palette) tag `Arc`s; the single
        // latent builds its own per-BAND tags here (n_palette entries per
        // head), because its bands genuinely differ: the nope bands carry the
        // writer format (FP8 E4M3 for the reference config) and the rope tail
        // bands are pinned BF16 — exactly the two arena runs allocated above.
        // K≡V, so both sides share one vec.
        let (k_fmt, v_fmt) = if single_latent {
            let nope_bands = crate::kv_cache::arena_table::LATENT_NOPE_BANDS.min(np);
            let on_gpu = matches!(self.inner.storage.default_location(), ArenaLocation::Gpu);
            let (k_active, _) =
                crate::kv_cache::active_kv_formats(self.inner.storage.k_format(), on_gpu);
            let nope_tag = ArenaFormatTag::from_kv_format(k_active).as_u8();
            let rope_tag = ArenaFormatTag::from_kv_format(KvFormat::Float(DType::BF16)).as_u8();
            let mut tags = Vec::with_capacity(np * self.inner.n_kv_head);
            for _h in 0..self.inner.n_kv_head {
                for p in 0..np {
                    tags.push(if p < nope_bands { nope_tag } else { rope_tag });
                }
            }
            let tags = std::sync::Arc::new(tags);
            (tags.clone(), tags)
        } else {
            (
                self.inner.active_k_fmt.clone(),
                self.inner.active_v_fmt.clone(),
            )
        };
        Ok(ChunkWindow {
            gids: HeadGids::from_vec(gids),
            usage,
            offset,
            k_pal: self.inner.identity_pal.clone(),
            v_pal: self.inner.identity_pal.clone(),
            k_scale: self.inner.identity_scale.clone(),
            v_scale: self.inner.identity_scale.clone(),
            k_fmt,
            v_fmt,
            // Fresh float writer chunk: transient, no resident record. The host
            // serializer builds per-forward scratch heads for it.
            meta: None,
            // Its Q is in its bands.
            q_sign: None,
        })
    }

    pub(in super::super) fn alloc_chunk_for_key(
        &self,
        key: super::super::arena::ArenaKey,
    ) -> Result<super::super::gid_pool::ChunkGid> {
        self.inner.alloc_chunk_for_key(key)
    }

    /// Create the arenas that mid-wave refusals asked this backing for.
    ///
    /// See [`BackingInner::create_deferred_arenas`]. Between forwards only, with
    /// no KV lock held — the top of a sealing pass.
    #[cfg(feature = "cuda")]
    pub fn create_deferred_arenas(&self) -> Result<usize> {
        self.inner.create_deferred_arenas()
    }

    /// Whether a refused pass is still owed an arena here — `false` once the
    /// gap between forwards has created every one asked for. What a pass that
    /// was turned away waits on before it tries again.
    pub fn arenas_owed(&self) -> bool {
        self.inner
            .deferred_arenas
            .lock()
            .is_ok_and(|p| !p.is_empty())
    }

    /// See [`BackingInner::alloc_chunk_run_for_key`].
    pub(in super::super) fn alloc_chunk_run_for_key(
        &self,
        key: super::super::arena::ArenaKey,
        len: usize,
    ) -> Result<Vec<super::super::gid_pool::ChunkGid>> {
        self.inner.alloc_chunk_run_for_key(key, len)
    }

    /// Bulk variant of [`Self::alloc_chunk_for_key`] — allocates `n`
    /// GIDs of the same `key` while paying the per-format pool mutex
    /// and the per-arena storage write lock only **once each**
    /// (instead of `n` times). Used by the cold-load
    /// `alloc_sealed_blocks_bulk` path where a single layer can need
    /// ~600 GIDs of the same format.
    pub(in super::super) fn alloc_chunks_for_key_bulk(
        &self,
        key: super::super::arena::ArenaKey,
        n: usize,
    ) -> Result<Vec<super::super::gid_pool::ChunkGid>> {
        self.inner.alloc_chunks_for_key_bulk(key, n)
    }

    pub(in super::super) fn ensure_arena_exists(
        &self,
        arena_idx: usize,
        key: super::super::arena::ArenaKey,
    ) -> Result<()> {
        self.inner.ensure_arena_exists(arena_idx, key)
    }
}
