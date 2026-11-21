//! Builds a backing and its layers, and fixes the geometry its storage uses.
//!
//! A new backing claims its device reservation up front, because that reservation
//! is the room the scheduler plans against. Single-latent mode must be set before
//! any chunk is allocated, and it drops writer arenas that were sized for the
//! grouped-query band width. New layers share storage and the arena table with
//! their parent, and each keeps its own block table.

use super::*;

impl ChunkedKvBacking {
    /// Whether `other` is a handle on this same backing — the same per-layer
    /// block table, so a batch index means the same sequence in both.
    pub(crate) fn shares_state_with(&self, other: &ChunkedKvBacking) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    pub(in super::super) fn set_block_gids_sharded_and_update_gpu(
        &self,
        batch_idx: usize,
        blk: usize,
        gids: super::super::head_gids::HeadGids,
        k_pal: std::sync::Arc<Vec<u8>>,
        v_pal: std::sync::Arc<Vec<u8>>,
        k_scale: std::sync::Arc<Vec<f32>>,
        v_scale: std::sync::Arc<Vec<f32>>,
        k_fmt: std::sync::Arc<Vec<u8>>,
        v_fmt: std::sync::Arc<Vec<u8>>,
        arena_info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
    ) -> Result<()> {
        self.inner.set_block_gids_sharded_and_update_gpu(
            self.layer_idx,
            batch_idx,
            blk,
            gids,
            k_pal,
            v_pal,
            k_scale,
            v_scale,
            k_fmt,
            v_fmt,
            arena_info,
        )
    }

    /// Create a new chunked backing with float storage.
    ///
    /// This is the legacy API for backward compatibility.
    /// For quantized storage, use [`new_with_format`](Self::new_with_format).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        initial_batch: usize,
        n_kv_head: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
        initial_max_seq_len: usize,
    ) -> Result<Self> {
        Self::new_with_format(
            initial_batch,
            n_kv_head,
            head_dim,
            KvFormat::Float(dtype),
            KvFormat::Float(dtype),
            device,
            initial_max_seq_len,
        )
    }

    /// Create a new chunked backing with specified storage format.
    ///
    /// K and V caches can use different formats. For quantized storage, the
    /// `head_dim` must be divisible by the block size (32).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_format(
        initial_batch: usize,
        n_kv_head: usize,
        head_dim: usize,
        k_format: KvFormat,
        v_format: KvFormat,
        device: &Device,
        initial_max_seq_len: usize,
    ) -> Result<Self> {
        Self::new_with_format_adaptive(
            initial_batch,
            n_kv_head,
            head_dim,
            k_format,
            v_format,
            device,
            initial_max_seq_len,
            None,
        )
    }

    /// Enable single-latent (K≡V) mode — one latent serves as both key and
    /// value (DeepSeek MLA-style). V band GIDs alias their K band GIDs and
    /// contiguous writes skip the V plane. Must be set before any chunk is
    /// allocated; applies to every layer sharing this backing group.
    pub fn set_single_latent(&self, on: bool) {
        self.inner
            .single_latent
            .store(on, std::sync::atomic::Ordering::Relaxed);
        // **No record stride to resize here any more.** It used to be cached on the meta
        // pool and had to be widened to match the new band count (single-latent = 8
        // bands, GQA = 4) before any chunk allocated, or a record would overrun its slot.
        // `build_meta_records` now derives the record size from the live `n_palette()` on
        // every call and picks the arena key from that, so flipping the flag above is by
        // itself enough and there is no second copy to keep in step.
        // The constructor's `keep_reserves` ran while single_latent was still
        // false, so it minted the writer/candidate arenas at the GQA band width
        // (`head_dim / N_PALETTE` = 128). The single latent needs them at its
        // own width (`head_dim / 16` = 32) — the width `resolve_arena_info` and
        // the band writes both assume. Nothing has allocated a chunk yet (this
        // runs immediately after construction, before any write), so drop the
        // mis-sized arenas and the reserves named at that width; the pool keeps
        // their registrations and the next allocation recreates each at the
        // single-latent width. GQA never calls this, so its reserves stand.
        if on {
            let _ = self.inner.storage.truncate_arenas(0);
            self.inner.pool.clear_reserves();
        }
    }

    /// Whether single-latent (K≡V) mode is active.
    pub fn single_latent(&self) -> bool {
        self.inner
            .single_latent
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Per-head band count for KvHead records: [`LATENT_N_BANDS`] on the
    /// single-latent path, [`N_PALETTE`] for GQA / palette4. Drives the KvHead
    /// record layout (host serializer + device accessors must agree).
    pub fn n_palette(&self) -> usize {
        if self.single_latent() {
            crate::kv_cache::arena_table::LATENT_N_BANDS
        } else {
            crate::kv_cache::arena_table::N_PALETTE
        }
    }

    /// Mark that a sealed chunk now carries a non-identity pal_map (sticky).
    pub fn set_mapped_sealed(&self) {
        self.inner
            .mapped_sealed
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether any sealed chunk carries a non-identity pal_map — attention
    /// launches must then use the MAPPED kernel instantiation.
    pub fn has_mapped_sealed(&self) -> bool {
        self.inner
            .mapped_sealed
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Create a new chunked backing with adaptive per-block format selection.
    ///
    /// The `k_format`/`v_format` serve as the ceiling (highest fidelity) format.
    ///
    /// Note: compression policy is owned by the session layer and only used
    /// here to warm the shared adaptive arenas during construction. Pass
    /// `None` to skip warming arenas for adaptive candidates.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_format_adaptive(
        initial_batch: usize,
        n_kv_head: usize,
        head_dim: usize,
        k_format: KvFormat,
        v_format: KvFormat,
        device: &Device,
        initial_max_seq_len: usize,
        compression: Option<CompressionPolicy>,
    ) -> Result<Self> {
        if initial_batch == 0 {
            candle::bail!("chunked backing initial_batch must be non-zero")
        }

        // For quantized storage, head_dim must be block-aligned (check both K and V)
        for (label, format) in [("K", &k_format), ("V", &v_format)] {
            if let KvFormat::Quantized(qf) = format {
                let block_size = qf.block_size();
                if !head_dim.is_multiple_of(block_size) {
                    candle::bail!(
                        "quantized {} KV cache requires head_dim ({}) to be divisible by block size ({})",
                        label,
                        head_dim,
                        block_size
                    )
                }
            }
        }

        let initial_max_blocks = cmp::max(1, initial_max_seq_len.div_ceil(CHUNK_SIZE));
        // Determine default location based on device
        let default_location = if device.is_cpu() {
            ArenaLocation::Cpu
        } else {
            ArenaLocation::Gpu
        };

        // Claim the device reservation here rather than on the first arena.
        // It is a startup cost (the balloon touches every granule it takes) and
        // this is the last moment before the scheduler starts asking how much
        // room there is — a question the free-region count answers, and which
        // has no answer at all until the reservation exists.
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device {
            super::super::region_pool::ensure(&cuda.cuda_stream())?;
        }

        let storage = ArenaStorage::new(k_format, v_format, default_location);

        let pool = super::super::gid_pool::ChunkGidPool::new();

        let pinned_stager = match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_dev) => PinnedStager::new(cuda_dev),
            _ => PinnedStager::new_from_device(device),
        };

        let inner = Arc::new(BackingInner {
            storage,
            state_registry: RwLock::new(ahash::HashMap::new()),
            pool,
            device: device.clone(),
            n_kv_head,
            head_dim,
            identity_pal: {
                use crate::kv_cache::arena_table::N_PALETTE;
                let pal_bytes = (head_dim / 4).max(1);
                let sub_hd = (head_dim / N_PALETTE).max(1);
                let mut buf = vec![0u8; n_kv_head * pal_bytes];
                for h in 0..n_kv_head {
                    let slice = &mut buf[h * pal_bytes..(h + 1) * pal_bytes];
                    for d in 0..head_dim {
                        let pal_idx = ((d / sub_hd).min(N_PALETTE - 1)) as u8;
                        slice[d / 4] |= pal_idx << ((d % 4) * 2);
                    }
                }
                Arc::new(buf)
            },
            identity_scale: {
                use crate::kv_cache::arena_table::N_PALETTE;
                Arc::new(vec![1.0f32; n_kv_head * N_PALETTE])
            },
            active_k_fmt: {
                use crate::kv_cache::arena_table::N_PALETTE;
                let on_gpu = default_location == ArenaLocation::Gpu;
                let (k, _) = crate::kv_cache::active_kv_formats(k_format, on_gpu);
                Arc::new(vec![
                    ArenaFormatTag::from_kv_format(k).as_u8();
                    n_kv_head * N_PALETTE
                ])
            },
            active_v_fmt: {
                use crate::kv_cache::arena_table::N_PALETTE;
                let on_gpu = default_location == ArenaLocation::Gpu;
                let (_, v) = crate::kv_cache::active_kv_formats(k_format, on_gpu);
                Arc::new(vec![
                    ArenaFormatTag::from_kv_format(v).as_u8();
                    n_kv_head * N_PALETTE
                ])
            },
            pinned_stager,
            meta_pool: super::super::meta_pool::MetaPool::new(device.clone()),
            single_latent: std::sync::atomic::AtomicBool::new(false),
            mapped_sealed: std::sync::atomic::AtomicBool::new(false),
            deferred_arenas: Mutex::new(Vec::new()),
            fresh_arenas: Mutex::new(Vec::new()),
        });

        // Register for cooperative compaction
        register_backing(&inner);

        let state = Arc::new(RwLock::new(BlockTableState::new(
            0,
            initial_max_blocks,
            initial_batch,
        )));
        register_state(&inner, &state);

        let backing = Self {
            inner,
            layer_idx: 0,
            state,
        };
        if !device.is_cpu() {
            backing.keep_reserves(compression.as_ref())?;
        }
        Ok(backing)
    }

    /// Create a new layer backing that shares storage, pool, and arena table
    /// with an existing backing. Each layer gets its own per-layer state.
    pub fn new_layer(
        &self,
        layer_idx: usize,
        initial_batch: usize,
        initial_max_seq_len: usize,
    ) -> Self {
        let initial_max_blocks = cmp::max(1, initial_max_seq_len.div_ceil(CHUNK_SIZE));
        let state = Arc::new(RwLock::new(BlockTableState::new(
            layer_idx,
            initial_max_blocks,
            initial_batch,
        )));
        register_state(&self.inner, &state);
        Self {
            inner: Arc::clone(&self.inner),
            state,
            layer_idx,
        }
    }

    /// Wait for all previously-enqueued background quantization work to complete.
    /// Clone the pinned stager for use by a sibling worker.
    pub fn pinned_stager(&self) -> candle::quantized::pinned_staging::PinnedStager {
        self.inner.pinned_stager.clone()
    }

    /// Number of KV heads.
    pub fn n_kv_head(&self) -> usize {
        self.inner.n_kv_head
    }

    /// Dimension of each head.
    pub fn head_dim(&self) -> usize {
        self.inner.head_dim
    }

    /// Elements one chunk slot holds at this backing's geometry — one
    /// `(head, palette-band, side)` of `CHUNK_SIZE` tokens.
    ///
    /// The number every size-class and payload-length question is asked with,
    /// exposed because callers outside this crate (the persistence gather)
    /// need it to turn a band's format tag into a byte length.
    pub fn elems_per_chunk(&self) -> usize {
        self.inner.elems_per_chunk()
    }

    /// Returns the K storage format for this backing.
    pub fn k_format(&self) -> KvFormat {
        self.inner.storage.k_format()
    }

    /// Returns the V storage format for this backing.
    pub fn v_format(&self) -> KvFormat {
        self.inner.storage.v_format()
    }

    /// Get the DType for this backing's default format, or None if quantized.
    ///
    /// For quantized storage, this returns None. Use `kv_format()` to get
    /// the full format including quantization type.
    pub fn dtype(&self) -> Option<DType> {
        self.inner.storage.dtype()
    }

    /// Returns true if using quantized storage.
    pub fn is_quantized(&self) -> bool {
        self.inner.storage.is_quantized()
    }
}
