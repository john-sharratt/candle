//! BackingInner and ChunkedKvBacking implementation.
//!
//! This module contains:
//! - `BackingInner` - The inner shared state
//! - `ChunkedKvBacking` - The main public API with constructors and accessors

use ahash::{HashMap, HashMapExt};
use candle::quantized::GgmlDType;
use std::cmp;
use std::sync::{Arc, Mutex, RwLock, RwLockWriteGuard, Weak};

use candle::quantized::pinned_staging::{Generation, PinnedStager};
use candle::quantized::QTensor;
#[cfg(feature = "cuda")]
use candle::quantized::{arena_compact_copy_async, CompactMove};
use candle::{DType, Device, Result, Tensor};

use super::{
    Arena, ArenaStorage, ArenaStorageState, BlockTableState, ChunkMeta, CompressionPolicy,
    SealedChunk, StoragePolicy,
};
use crate::kv_cache::arena_table::{ArenaLocation, PerHeadEntry};
use crate::kv_cache::{HeadGids, KvFormat, ResolvedArenaInfo};
use crate::{arena_gid_stride, CHUNK_SIZE};

/// Default fraction of reclaimable arenas that triggers pack-down.
const DEFAULT_DEFRAG_THRESHOLD: f32 = 0.20;

/// Global registry of all ChunkedKvBacking instances for cooperative compaction.
/// When one backing needs to grow arenas but fails (OOM), it can ask others to compact.
static BACKING_REGISTRY: Mutex<Vec<Weak<BackingInner>>> = Mutex::new(Vec::new());

/// Register a backing for cooperative compaction.
pub(super) fn register_backing(inner: &Arc<BackingInner>) {
    if let Ok(mut registry) = BACKING_REGISTRY.lock() {
        // Clean up dead entries while we're here
        registry.retain(|w| w.strong_count() > 0);
        registry.push(Arc::downgrade(inner));
    }
}

fn register_state(inner: &Arc<BackingInner>, state: &Arc<RwLock<BlockTableState>>) {
    if let Ok(mut registry) = inner.state_registry.write() {
        registry.retain(|_, w| w.strong_count() > 0);
        registry.insert(state.read().unwrap().layer_idx, Arc::downgrade(state));
    }
}

/// Ask all other backings to compact, returns total chunks freed.
pub(super) fn request_global_compact() -> usize {
    let mut freed = 0;
    if let Ok(registry) = BACKING_REGISTRY.lock() {
        for weak in registry.iter() {
            if let Some(backing) = weak.upgrade() {
                if let Ok(n) = backing.compact_arenas() {
                    freed += n;
                }
            }
        }
    }
    freed
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
    state_registry: RwLock<ahash::HashMap<usize, Weak<RwLock<BlockTableState>>>>,
    /// Shared chunk GID allocation pool across all layers.
    pub(crate) pool: super::gid_pool::ChunkGidPool,
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
    /// Reusable pinned-memory stager for async H2D metadata uploads.
    pub(crate) pinned_stager: PinnedStager,
}

impl BackingInner {
    fn needs_compaction(&self, fragmentation_threshold: f32) -> Result<bool> {
        if self.pool.has_reclaimable() || self.pool.needs_defragmentation(fragmentation_threshold) {
            return Ok(true);
        }

        let max_used_chunk = self.pool.max_gid().unwrap_or(-1);
        let needed_arenas = if max_used_chunk < 0 {
            0
        } else {
            ((max_used_chunk as usize) / arena_gid_stride()) + 1
        };
        Ok(needed_arenas < self.storage.arena_count()?)
    }

    #[allow(dead_code)]
    fn registered_states(&self) -> Vec<Arc<RwLock<BlockTableState>>> {
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

    #[allow(dead_code)]
    fn apply_gid_remap(
        states: &mut [RwLockWriteGuard<'_, BlockTableState>],
        remap: &HashMap<i64, super::gid_pool::ChunkGid>,
    ) -> Result<()> {
        if remap.is_empty() {
            return Ok(());
        }

        for state in states.iter_mut() {
            for seq in state.sequences.iter_mut().flatten() {
                let mut changed_seq = false;
                let block_count = seq.block_count();
                for blk in 0..block_count {
                    // GID remap during arena compaction: only the raw GID values
                    // change (chunks moved to different physical slots within the
                    // same arena format). The encoded byte semantics are
                    // unchanged, so pal/scale must be preserved verbatim.
                    let replacement = seq.chunk_at(blk).and_then(|cw| {
                        let mut new_gids = cw.gids.clone();
                        let mut changed = false;
                        for gid in new_gids.iter_mut() {
                            if let Some(new_gid) = remap.get(&gid.raw()) {
                                *gid = new_gid.clone();
                                changed = true;
                            }
                        }
                        if changed {
                            Some((
                                new_gids,
                                cw.k_pal.clone(),
                                cw.v_pal.clone(),
                                cw.k_scale.clone(),
                                cw.v_scale.clone(),
                            ))
                        } else {
                            None
                        }
                    });
                    if let Some((new_gids, k_pal, v_pal, k_scale, v_scale)) = replacement {
                        seq.set_block_gids(blk, new_gids, k_pal, v_pal, k_scale, v_scale);
                        changed_seq = true;
                    }
                }
                if changed_seq {
                    seq.invalidate_gpu_chunks();
                }
            }
        }
        Ok(())
    }

    fn release_empty_arenas(&self) -> Result<usize> {
        let mut freed = 0;

        // Phase 1: Pool-driven tombstoning of fully-free arenas.
        // For each format key, repeatedly call next_tombstone until exhausted.
        let keys = self.pool.format_keys();
        for key in keys {
            while let Some(arena_idx) = self.pool.next_tombstone(key.clone()) {
                self.storage.release_arena(arena_idx)?;
                freed += 1;
            }
        }

        // Phase 2: Truncate unused tail arenas (tombstoned or otherwise).
        // Use the pool's registry to find the highest live arena.
        let max_used_chunk = self.pool.max_gid().unwrap_or(-1);

        let needed_arenas = if max_used_chunk < 0 {
            0
        } else {
            ((max_used_chunk as usize) / arena_gid_stride()) + 1
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
        Ok(freed)
    }

    /// Defragment GPU arenas by greedily draining the least-occupied arenas for
    /// each format key into free slots elsewhere, then patching all host-side
    /// GID references in one batched pass.
    ///
    /// Only chunks that must leave an arena being freed are ever touched —
    /// holes inside kept arenas are left in place.
    ///
    /// All moves across all keys are batched into a single GPU copy + sync +
    /// remap so the expensive per-call overhead (device sync, state lock walk)
    /// is paid once regardless of how many arenas are drained.
    ///
    /// Every failure mode degrades gracefully: allocated destination GIDs are
    /// dropped (auto-returned to the pool) and the batch carries on without
    /// them.  The only unrecoverable path is a successful GPU copy followed by
    /// a remap failure, which would leave host and device state diverged.
    pub(crate) fn defragment_arenas(&self, fragmentation_threshold: f32) -> Result<usize> {
        if !self.pool.needs_defragmentation(fragmentation_threshold) {
            return Ok(0);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = fragmentation_threshold;
            return Ok(0);
        }

        #[cfg(feature = "cuda")]
        {
            let Device::Cuda(cuda_dev) = &self.device else {
                return Ok(0);
            };

            // Phase 1 — build the full move batch across all keys.
            //
            // `all_dst_gids` keeps destination ChunkGids alive until after
            // apply_gid_remap stores them in sequences; entries are parallel to
            // the `remap` values (remap holds clones, all_dst_gids the originals).
            let mut all_moves: Vec<CompactMove> = Vec::new();
            let mut all_dst_gids: Vec<super::gid_pool::ChunkGid> = Vec::new();
            let mut remap: HashMap<i64, super::gid_pool::ChunkGid> = HashMap::new();

            // Hold a single storage read lock across the entire batch build.
            // Inside, every (base_ptr, stride) per arena is resolved at most once
            // and reused — chunk_copy_span is a fixed (base + chunk_idx * stride),
            // so a per-arena cache lets the inner loop be pure pointer math.
            self.storage.read(|state| {
                let mut arena_span: ahash::HashMap<usize, (u64, u32)> = ahash::HashMap::new();
                let resolve_span = |ai: usize,
                                    cache: &mut ahash::HashMap<usize, (u64, u32)>|
                 -> Option<(u64, u32)> {
                    if let Some(&v) = cache.get(&ai) {
                        return Some(v);
                    }
                    let span = state.arena(ai).and_then(|a| a.chunk_copy_span(0))?;
                    cache.insert(ai, span);
                    Some(span)
                };

                for key in self.pool.format_keys() {
                    if self.pool.defragmentable_ratio_for(&key) <= fragmentation_threshold {
                        continue;
                    }

                    // Arenas sorted emptiest-first: process in that order so we
                    // drain the cheapest targets and stop when no space remains.
                    let arenas = self.pool.arenas_sorted_by_live_for_key(&key);
                    if arenas.len() < 2 {
                        continue;
                    }

                    for &(target_arena, live_count) in &arenas {
                        if live_count == 0 {
                            // Already empty; release_empty_arenas handles it.
                            continue;
                        }

                        // Live GIDs come from pool state — no sequence scan needed.
                        let live_gids = self.pool.live_gids_for_arena(target_arena);
                        if live_gids.is_empty() {
                            continue;
                        }

                        // Allocate replacement GIDs, strictly excluding target_arena.
                        let slice_start = all_dst_gids.len();
                        let mut alloc_ok = true;
                        for _ in 0..live_gids.len() {
                            match self.pool.allocate_for_excluding(key.clone(), target_arena) {
                                Some(gid) => all_dst_gids.push(gid),
                                None => {
                                    alloc_ok = false;
                                    break;
                                }
                            }
                        }
                        if !alloc_ok {
                            all_dst_gids.truncate(slice_start);
                            break;
                        }

                        // Resolve the src arena once — every live_gid lives in target_arena.
                        let Some((src_base, src_stride)) =
                            resolve_span(target_arena, &mut arena_span)
                        else {
                            all_dst_gids.truncate(slice_start);
                            break;
                        };
                        if src_stride % 16 != 0 {
                            all_dst_gids.truncate(slice_start);
                            break;
                        }

                        // Build CUDA copy descriptors from cached pointer arithmetic.
                        let moves_start = all_moves.len();
                        let mut build_ok = true;
                        for (src_raw, dst_gid) in
                            live_gids.iter().zip(all_dst_gids[slice_start..].iter())
                        {
                            let src_ci = (*src_raw as usize) % arena_gid_stride();
                            let dst_ai = dst_gid.arena_idx();
                            let Some((dst_base, dst_stride)) =
                                resolve_span(dst_ai, &mut arena_span)
                            else {
                                build_ok = false;
                                break;
                            };
                            if dst_stride != src_stride {
                                build_ok = false;
                                break;
                            }
                            all_moves.push(CompactMove {
                                dst: dst_base + dst_gid.chunk_idx() as u64 * dst_stride as u64,
                                src: src_base + src_ci as u64 * src_stride as u64,
                                stride_bytes: src_stride,
                                _pad: 0,
                            });
                        }
                        if !build_ok {
                            all_moves.truncate(moves_start);
                            all_dst_gids.truncate(slice_start);
                            break;
                        }

                        // Commit this arena's moves into the remap.
                        for (&src_raw, dst_gid) in
                            live_gids.iter().zip(all_dst_gids[slice_start..].iter())
                        {
                            remap.insert(src_raw, dst_gid.clone());
                        }
                    }
                }
            })?;

            if all_moves.is_empty() {
                return Ok(0);
            }

            // Phase 2 — one lock, one GPU copy, one sync, one remap.
            let state_arcs = self.registered_states();
            let mut locked_states: Vec<RwLockWriteGuard<'_, BlockTableState>> =
                Vec::with_capacity(state_arcs.len());
            let mut lock_ok = true;
            for sa in &state_arcs {
                match sa.write() {
                    Ok(g) => locked_states.push(g),
                    Err(_) => {
                        lock_ok = false;
                        break;
                    }
                }
            }
            if !lock_ok {
                // remap + all_dst_gids auto-freed on drop.
                return Ok(0);
            }

            // On copy failure the source data is still intact (remap not yet
            // applied), so all sequence references remain valid.
            // remap + all_dst_gids are freed by drop.
            // No device sync needed: the CUDA stream serialises the copy before
            // any subsequent kernel that touches the same slots.
            let _generation = self.pinned_stager.begin_generation();
            let primary_stream = cuda_dev.cuda_stream();
            if arena_compact_copy_async(&all_moves, 128, &primary_stream, &self.pinned_stager)
                .is_err()
            {
                return Ok(0);
            }

            // Replaced source ChunkGids are dropped by set_block_gids,
            // auto-returning them to the pool and emptying the drained arenas.
            // After this point GPU data has moved; a remap failure would leave
            // host references stale — propagate it upward.
            if let Err(e) = Self::apply_gid_remap(&mut locked_states, &remap) {
                return Err(e);
            }
            self.pool.resync_counters();
            Ok(remap.len())
        }
    }

    /// Compact arenas by first pack-down defragmenting when worthwhile, then
    /// tombstoning any empty middle arenas and truncating unused tails.
    pub(crate) fn compact_arenas(&self) -> Result<usize> {
        let _ = self.defragment_arenas(DEFAULT_DEFRAG_THRESHOLD)?;
        self.release_empty_arenas()
    }
}

/// Shared backing storage for a chunked (paged) KV cache.
///
/// Matches the addressing expected by the CUDA chunked ragged-attention kernels.
/// Supports both standard float storage and block-quantized (Q4_0, Q8_0) storage.
///
/// Multiple layers share the same `Arc<BackingInner>` (storage, pool)
/// but each layer has its own per-layer `state` (sequences, max_blocks).
#[derive(Debug, Clone)]
pub struct ChunkedKvBacking {
    pub(super) inner: Arc<BackingInner>,
    pub(crate) layer_idx: usize,
    /// Per-layer block table state (sequences, max_blocks).
    pub(crate) state: Arc<RwLock<BlockTableState>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeGpuChunkSyncStats {
    pub rebuilds: u64,
    pub reuses: u64,
    pub empty: u64,
    pub rebuild_time: std::time::Duration,
    pub reuse_time: std::time::Duration,
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
    pub(super) fn set_block_gids_sharded_and_update_gpu(
        &self,
        layer_idx: usize,
        batch_idx: usize,
        block_idx: usize,
        gids: HeadGids,
        k_pal: std::sync::Arc<Vec<u8>>,
        v_pal: std::sync::Arc<Vec<u8>>,
        k_scale: std::sync::Arc<Vec<f32>>,
        v_scale: std::sync::Arc<Vec<f32>>,
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
            slot.set_block_gids(block_idx, gids, k_pal, v_pal, k_scale, v_scale);
            slot.update_gpu_chunk(block_idx, self.n_kv_head, self.head_dim, arena_info)?;
        }
        Ok(())
    }
}

impl ChunkedKvBacking {
    pub(super) fn set_block_gids_sharded_and_update_gpu(
        &self,
        batch_idx: usize,
        blk: usize,
        gids: super::head_gids::HeadGids,
        k_pal: std::sync::Arc<Vec<u8>>,
        v_pal: std::sync::Arc<Vec<u8>>,
        k_scale: std::sync::Arc<Vec<f32>>,
        v_scale: std::sync::Arc<Vec<f32>>,
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

        let storage = ArenaStorage::new(k_format, v_format, default_location);

        let pool = super::gid_pool::ChunkGidPool::new();

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
            pinned_stager,
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
            backing.warm_protected_arenas(compression.as_ref())?;
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

    /// Returns true if this backing has enough reclaimable/tombstoned work to
    /// justify a compaction pass.
    pub fn needs_compaction(&self) -> Result<bool> {
        self.inner.needs_compaction(DEFAULT_DEFRAG_THRESHOLD)
    }

    /// Defragment and compact the backing when the reclaimable-arena ratio for
    /// any key exceeds `fragmentation_threshold`.
    pub fn defragment(&self, fragmentation_threshold: f32) -> Result<usize> {
        let _ = self.inner.defragment_arenas(fragmentation_threshold)?;
        self.inner.release_empty_arenas()
    }

    /// Compact arenas by using the default defrag threshold, then releasing
    /// unused middle and tail arenas.
    /// Returns the number of arenas freed.
    pub fn compact(&self) -> Result<usize> {
        self.inner.compact_arenas()
    }

    /// Get the current batch capacity (number of sequence slots).
    pub fn batch_capacity(&self) -> usize {
        self.state
            .read()
            .expect("chunked state lock poisoned")
            .sequences
            .len()
    }

    /// Grow the batch capacity to accommodate more sequences.
    ///
    /// This expands the block table and slots vector to hold `new_capacity` sequences.
    /// Existing sequences are preserved.
    pub fn grow_batch_capacity(&self, new_capacity: usize) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        if new_capacity <= state.sequences.len() {
            return Ok(());
        }

        // Grow the slots vector (block table is derived from slots on demand)
        state.sequences.resize(new_capacity, None);

        Ok(())
    }

    /// Number of KV heads.
    pub fn n_kv_head(&self) -> usize {
        self.inner.n_kv_head
    }

    /// Dimension of each head.
    pub fn head_dim(&self) -> usize {
        self.inner.head_dim
    }

    /// Begin a stager generation guard.
    ///
    /// While the returned [`Generation`] is alive, the pinned arena will not be
    /// reset — all device-mapped pointers from `submit()` remain valid.  When
    /// the last generation drops, the stream is synchronised and arenas are
    /// reclaimed.
    ///
    /// Returns `None` on non-CUDA devices or if no stager is configured.
    pub fn begin_stager_generation(&self) -> Option<Generation> {
        Some(self.inner.pinned_stager.begin_generation())
    }

    /// Begin a stager generation, panicking if no stager is configured.
    ///
    /// Use this when a Generation is mandatory (e.g. during forward passes
    /// that will run quantization kernels).
    pub fn begin_stager_generation_required(&self) -> Generation {
        self.inner.pinned_stager.begin_generation()
    }

    /// Returns a snapshot of the live chunks for a sequence as [`SealedChunk`]s.
    ///
    /// Used by the slot pool system to build `SlotStateHost` entries from current
    /// chunk data without waiting for a full turn seal.
    ///
    /// No positional state is captured — RoPE is applied at the
    /// latest responsible moment by the attention kernel using a
    /// `slice_rope` recomputed from the destination slot's cumulative
    /// usage.  See [`SealedChunk`] docs.
    ///
    /// Returns `None` if the sequence slot is not allocated.
    /// First writer-owned chunk index for the given sequence slot.
    /// Returns `None` if the slot is unallocated.  See
    /// [`super::types::SequenceState::writer_start_idx`].
    pub fn writer_start_idx_for_seq(&self, batch_idx: usize) -> Option<usize> {
        let state = self.state.read().ok()?;
        let seq = state.sequences.get(batch_idx)?.as_ref()?;
        Some(seq.writer_start_idx())
    }

    pub fn live_chunks_as_sealed(
        &self,
        batch_idx: usize,
        arena_infos: &[ResolvedArenaInfo],
    ) -> Option<Vec<SealedChunk>> {
        let state = self.state.read().ok()?;
        let seq = state.sequences.get(batch_idx)?.as_ref()?;
        let chunks: Vec<SealedChunk> = seq
            .chunks_slice()
            .iter()
            .map(|cw| {
                let byte_size = cw.gids.arena_byte_size(arena_infos);
                SealedChunk {
                    gids: cw.gids.clone(),
                    offset: cw.offset,
                    token_count: cw.usage as u16,
                    k_pal: cw.k_pal.clone(),
                    v_pal: cw.v_pal.clone(),
                    k_scale: cw.k_scale.clone(),
                    v_scale: cw.v_scale.clone(),
                    byte_size,
                }
            })
            .collect();
        Some(chunks)
    }

    /// Rebuild the GPU slot-state buffers for all sequences in `batch_entries`
    /// and return `(device_ptr, n_chunks, write_chunk_idx)` per entry.
    ///
    /// Invalidate the persistent GPU slot-state buffers for all sequences in
    /// `batch_entries`.
    ///
    /// Explicitly clear the persistent decode GPU buffers for the selected
    /// sequences so the next decode sync is forced to rebuild them.
    pub fn invalidate_decode_gpu_chunks(&self, batch_entries: &[(usize, usize)]) {
        if let Ok(mut state) = self.state.write() {
            for &(seq_idx, _) in batch_entries {
                if let Some(Some(seq)) = state.sequences.get_mut(seq_idx) {
                    seq.invalidate_gpu_chunks();
                }
            }
        }
    }

    /// Lightweight host-side validation for the paged decode hot path.
    ///
    /// This is intended to catch full or stale tail-slice metadata before the
    /// CUDA kernel is launched, while keeping the check effectively O(batch).
    pub fn validate_decode_batch_state(&self, batch_entries: &[(usize, usize)]) -> Result<()> {
        if batch_entries.is_empty() {
            return Ok(());
        }
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        for &(seq_idx, seq_offset) in batch_entries {
            let seq = state
                .sequences
                .get(seq_idx)
                .and_then(|s| s.as_ref())
                .ok_or_else(|| {
                    let cap = state.sequences.len();
                    let alloc_map: Vec<String> = state
                        .sequences
                        .iter()
                        .enumerate()
                        .map(|(i, s)| format!("{}={}", i, if s.is_some() { "S" } else { "_" }))
                        .collect();
                    candle::Error::Msg(format!(
                        "chunked decode validation failed for batch_idx {}: sequence slot is not allocated (capacity={}, entries={:?}, slots=[{}])",
                        seq_idx,
                        cap,
                        batch_entries,
                        alloc_map.join(",")
                    ))
                })?;
            seq.validate_decode_state(seq_idx, seq_offset)?;
        }
        Ok(())
    }

    /// Check that no two sessions in `batch_entries` share a raw GID value for
    /// any K or V head/palette slot.  Shared GIDs across sessions indicate
    /// aliased arena storage and will cause cross-session KV contamination.
    ///
    /// Returns a list of human-readable violation strings (empty = clean).
    /// Each entry reports the GID, computed GPU pointer, and both owner locations.
    pub fn validate_gid_uniqueness(
        &self,
        batch_entries: &[(usize, usize)],
    ) -> candle::Result<Vec<String>> {
        use crate::kv_cache::arena_table::N_PALETTE;
        use std::collections::HashMap;

        if batch_entries.is_empty() {
            return Ok(Vec::new());
        }

        // Resolve arena info outside the state lock.
        let arena_info = self.resolve_arena_info()?;

        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let n_kv_head = self.inner.n_kv_head;

        // Maps raw GID → (batch_idx, blk, head, pal, is_k)
        let mut seen: HashMap<i64, (usize, usize, usize, usize, bool)> = HashMap::new();
        let mut violations: Vec<String> = Vec::new();

        for &(batch_idx, _) in batch_entries {
            let seq = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                Some(s) => s,
                None => continue,
            };
            for blk in 0..seq.block_count() {
                let cw = match seq.chunk_at(blk) {
                    Some(c) => c,
                    None => continue,
                };
                for h in 0..n_kv_head {
                    for p in 0..N_PALETTE {
                        let k_gid = cw.gids.k_gid_pal(h, p);
                        let v_gid = cw.gids.v_gid_pal(h, p);

                        let check =
                            |raw: i64,
                             is_k: bool,
                             seen: &mut HashMap<i64, (usize, usize, usize, usize, bool)>,
                             violations: &mut Vec<String>| {
                                let stride = arena_gid_stride();
                                let ptr = arena_info
                                    .get(raw as usize / stride)
                                    .map(|ai| {
                                        ai.base_ptr
                                            + (raw as usize % stride) as u64
                                                * ai.chunk_byte_stride as u64
                                    })
                                    .unwrap_or(0);

                                if let Some(&(ob, obl, oh, op, oik)) = seen.get(&raw) {
                                    if ob != batch_idx {
                                        violations.push(format!(
                                            "{}-GID ALIAS: gid={raw} ptr={ptr:#x} \
                                         → session {batch_idx} blk={blk} h={h} p={p} \
                                         vs session {ob} blk={obl} h={oh} p={op} {}",
                                            if is_k { "K" } else { "V" },
                                            if oik { "(K)" } else { "(V)" },
                                        ));
                                    }
                                } else {
                                    seen.insert(raw, (batch_idx, blk, h, p, is_k));
                                }
                            };

                        check(k_gid.raw(), true, &mut seen, &mut violations);
                        check(v_gid.raw(), false, &mut seen, &mut violations);
                    }
                }
            }
        }

        Ok(violations)
    }

    /// Synchronise decode slot-state buffers for the selected sequences and
    /// return both the raw slot pointers and aggregated rebuild-versus-reuse
    /// timing stats.
    pub fn sync_decode_gpu_chunks(
        &self,
        batch_entries: &[(usize, usize)],
        arena_info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
    ) -> candle::Result<(Vec<(u64, u32, u32)>, DecodeGpuChunkSyncStats)> {
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let mut results = Vec::with_capacity(batch_entries.len());
        let mut stats = DecodeGpuChunkSyncStats::default();
        for &(seq_idx, seq_offset) in batch_entries {
            let t_sync = std::time::Instant::now();
            let (result, sync_kind) = if let Some(Some(seq)) = state.sequences.get_mut(seq_idx) {
                seq.validate_decode_state(seq_idx, seq_offset)?;
                seq.sync_decode_gpu_chunks(n_kv_head, head_dim, seq_offset, arena_info)?
            } else {
                ((0, 0, 0), super::types::DecodeGpuChunksSyncKind::Empty)
            };
            let elapsed = t_sync.elapsed();
            match sync_kind {
                super::types::DecodeGpuChunksSyncKind::Rebuild => {
                    stats.rebuilds += 1;
                    stats.rebuild_time += elapsed;
                }
                super::types::DecodeGpuChunksSyncKind::Reuse => {
                    stats.reuses += 1;
                    stats.reuse_time += elapsed;
                }
                super::types::DecodeGpuChunksSyncKind::Empty => {
                    stats.empty += 1;
                }
            }
            results.push(result);
        }
        Ok((results, stats))
    }

    /// Returns the storage policy for sealed chunks.
    ///
    /// This determines the target format for sealed chunks after reconcile().
    pub fn storage_policy(&self) -> StoragePolicy {
        let k_format = self.inner.storage.k_format();
        let v_format = self.inner.storage.v_format();
        let location = self.inner.storage.default_location();
        match (location, k_format, v_format) {
            (ArenaLocation::Gpu, KvFormat::Float(dt), KvFormat::Float(_)) => {
                StoragePolicy::GpuFloat(dt)
            }
            (ArenaLocation::Gpu, KvFormat::Quantized(kf), KvFormat::Quantized(_)) => {
                StoragePolicy::GpuQuant(kf)
            }
            (ArenaLocation::Cpu, KvFormat::Float(dt), KvFormat::Float(_)) => {
                StoragePolicy::CpuFloat(dt)
            }
            (ArenaLocation::Cpu, KvFormat::Quantized(kf), KvFormat::Quantized(_)) => {
                StoragePolicy::CpuQuant(kf)
            }
            // Mixed float/quant — treat as quant with the quant format
            (ArenaLocation::Gpu, KvFormat::Quantized(kf), KvFormat::Float(_)) => {
                StoragePolicy::GpuQuant(kf)
            }
            (ArenaLocation::Gpu, KvFormat::Float(_), KvFormat::Quantized(vf)) => {
                StoragePolicy::GpuQuant(vf)
            }
            (ArenaLocation::Cpu, KvFormat::Quantized(kf), KvFormat::Float(_)) => {
                StoragePolicy::CpuQuant(kf)
            }
            (ArenaLocation::Cpu, KvFormat::Float(_), KvFormat::Quantized(vf)) => {
                StoragePolicy::CpuQuant(vf)
            }
        }
    }

    /// Returns the K storage format for this backing.
    pub fn k_format(&self) -> KvFormat {
        self.inner.storage.k_format()
    }

    /// Returns the V storage format for this backing.
    pub fn v_format(&self) -> KvFormat {
        self.inner.storage.v_format()
    }

    /// Returns the actual K/V format tags from the first live arena.
    ///
    /// Falls back to the backing's configured defaults if no arenas exist.
    /// Use this for kernel dispatch rather than `k_format()`/`v_format()` — the
    /// backing default may be a quantized target while arenas are still float.
    pub fn actual_kv_format_tags(
        &self,
    ) -> (
        crate::kv_cache::ArenaFormatTag,
        crate::kv_cache::ArenaFormatTag,
    ) {
        self.inner.storage.actual_kv_format_tags()
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

    /// Build the per-head table tensor fresh from current arena storage.
    pub fn per_head_table_sync(&self) -> Result<Tensor> {
        self.inner.per_head_table_sync()
    }
}

impl BackingInner {
    /// Build the per-head table tensor fresh from current arena storage.
    ///
    /// Returns shape `(num_arenas * n_kv_head, 7)` i64.
    /// Built on every attention call by scanning the active sequence block table —
    /// no persistent state to keep in sync.
    ///
    /// For each arena referenced by any active block across all sequences, one
    /// `PerHeadEntry` row is emitted per KV head.  Byte offsets and strides are
    /// computed from first principles for the combined K+V buffer layout:
    ///
    ///   chunk slot = [K_head_0 … K_head_{N-1}, V_head_0 … V_head_{N-1}]
    ///
    /// When per-head quantization is active and different heads use different
    /// arenas, the GID scan automatically covers all referenced arenas.
    pub fn per_head_table_sync(&self) -> Result<Tensor> {
        // ⚠️  CP3 COLLAPSE POINT — READ docs/kv_cache_unification.md §7.6, §11.4 FIRST  ⚠️
        //
        // With palette4, this table uses N_PALETTE=4 sub-entries per (arena, head) row.
        // GPU tensor shape: (num_arenas * n_kv_head, 28).
        // Indexed by pal0_arena_idx * n_kv_head + head_idx.
        //
        // All N_PALETTE sub-entries in a row are IDENTICAL for now (same arena, same fmt),
        // since all palette GIDs for a given (head, block) land in the same physical arena.
        // The per-palette chunk_idx varies and is carried separately in head_gids.
        //
        // chunk_byte_stride uses head_dim/N_PALETTE (each chunk is one palette sub-band).
        //
        // ⚠️  DO NOT add fields to BackingInner to solve this — fix arena allocation  ⚠️
        use crate::kv_cache::arena_table::{PaletteSubEntry, N_PALETTE};

        let n_kv_head = self.n_kv_head;
        let chunk_size = CHUNK_SIZE;
        let sub_head_dim = (self.head_dim / N_PALETTE).max(1);
        let device = &self.device;

        // Scan all active sequence blocks to find the highest arena index in use.
        // The kernel indexes the table as `table[arena_idx * n_kv_head + h]`, so
        // the table must be dense over [0, num_arenas).
        let num_arenas = {
            let states: Vec<Arc<RwLock<BlockTableState>>> = {
                let registry = self
                    .state_registry
                    .read()
                    .unwrap_or_else(|e| e.into_inner());
                registry.values().filter_map(|w| w.upgrade()).collect()
            };
            states
                .iter()
                .filter_map(|state| {
                    let guard = state.read().ok()?;
                    guard
                        .sequences
                        .iter()
                        .flatten()
                        .flat_map(|seq| seq.chunks_slice().iter())
                        .flat_map(|cw| cw.gids.iter())
                        .map(|gid| gid.arena_idx())
                        .max()
                })
                .max()
                .map(|m| m + 1)
                .unwrap_or(0)
        };

        if num_arenas == 0 {
            return Tensor::zeros((1, PerHeadEntry::COLS), candle::DType::I64, device);
        }

        self.storage.read(|s| {
            let arenas = s.arenas();
            let mut data: Vec<i64> = vec![0i64; num_arenas * n_kv_head * PerHeadEntry::COLS];

            for arena_idx in 0..num_arenas {
                let arena = match arenas.get(&arena_idx) {
                    Some(a) => a,
                    None => continue,
                };

                // Base pointer (0 for CPU arenas) — to_arena_entry handles cfg(cuda).
                let ae = arena.to_arena_entry();
                let base_ptr = ae.k_ptr; // k_ptr == v_ptr for the combined K+V buffer
                let k_tag = ae.k_format_tag;
                let v_tag = ae.v_format_tag;

                // Per-palette chunk byte stride: each chunk covers CHUNK_SIZE × sub_head_dim.
                // sub_head_dim = head_dim / N_PALETTE for palette4 sub-arenas.
                let chunk_byte_stride = match arena.format() {
                    KvFormat::Float(dtype) => {
                        (chunk_size * sub_head_dim) as i64 * dtype.size_in_bytes() as i64
                    }
                    KvFormat::Quantized(qfmt) => {
                        let q_ggml = qfmt.to_ggml_dtype();
                        (chunk_size * sub_head_dim / q_ggml.block_size()) as i64
                            * q_ggml.type_size() as i64
                    }
                };

                // Build one PaletteSubEntry; all N_PALETTE slots are identical
                // since every palette GID for this (arena, head) references the same arena.
                let sub = PaletteSubEntry {
                    k_ptr: base_ptr,
                    v_ptr: base_ptr,
                    k_byte_offset: 0,
                    v_byte_offset: 0,
                    k_chunk_byte_stride: chunk_byte_stride,
                    v_chunk_byte_stride: chunk_byte_stride,
                    k_format_tag: k_tag,
                    v_format_tag: v_tag,
                    k_outer_scale: 1.0,
                    v_outer_scale: 1.0,
                };
                let entry = PerHeadEntry::uniform(sub);
                let cols = entry.to_tensor_row();

                let row_base = arena_idx * n_kv_head;
                for h in 0..n_kv_head {
                    let offset = (row_base + h) * PerHeadEntry::COLS;
                    data[offset..offset + PerHeadEntry::COLS].copy_from_slice(&cols);
                }
            }

            Tensor::from_vec(data, (num_arenas * n_kv_head, PerHeadEntry::COLS), device)
        })?
    }

    /// Build a lightweight per-arena info snapshot for constructing persistent
    /// GPU slot buffers without building a full GPU tensor.
    ///
    /// Returns one [`ResolvedArenaInfo`] per arena index (dense, 0..num_arenas).
    /// The `num_arenas` count is determined by scanning all active GIDs.
    pub fn resolve_arena_info(&self) -> Result<Vec<ResolvedArenaInfo>> {
        use crate::kv_cache::arena_table::{ArenaFormatTag, ResolvedArenaInfo, N_PALETTE};

        let chunk_size = CHUNK_SIZE;
        let sub_head_dim = (self.head_dim / N_PALETTE).max(1);

        self.storage.read(|s| {
            let arenas = s.arenas();
            if arenas.is_empty() {
                return Ok(Vec::new());
            }
            let num_arenas = arenas.keys().max().copied().unwrap_or(0) + 1;
            let mut info = vec![
                ResolvedArenaInfo {
                    base_ptr: 0,
                    chunk_byte_stride: 0,
                    k_format_tag: ArenaFormatTag::BF16,
                    v_format_tag: ArenaFormatTag::BF16,
                };
                num_arenas
            ];

            for (&arena_idx, arena) in arenas.iter() {
                let ae = arena.to_arena_entry();
                let base_ptr = ae.k_ptr;
                let k_tag = ae.k_format_tag;
                let v_tag = ae.v_format_tag;

                let chunk_byte_stride = match arena.format() {
                    KvFormat::Float(dtype) => {
                        (chunk_size * sub_head_dim) as i64 * dtype.size_in_bytes() as i64
                    }
                    KvFormat::Quantized(qfmt) => {
                        let q_ggml = qfmt.to_ggml_dtype();
                        (chunk_size * sub_head_dim / q_ggml.block_size()) as i64
                            * q_ggml.type_size() as i64
                    }
                };

                info[arena_idx] = ResolvedArenaInfo {
                    base_ptr,
                    chunk_byte_stride,
                    k_format_tag: k_tag,
                    v_format_tag: v_tag,
                };
            }

            Ok(info)
        })?
    }
}

impl ChunkedKvBacking {
    pub fn resolve_arena_info(&self) -> Result<Vec<ResolvedArenaInfo>> {
        self.inner.resolve_arena_info()
    }

    /// Get the number of arenas in backing storage.
    pub fn arena_count(&self) -> Result<usize> {
        self.inner.storage.arena_count()
    }

    /// Count the number of quantized arenas (excluding R16 raw storage).
    ///
    /// Returns (quantized_count, total_count) tuple.
    /// Useful for validating that quantization is actually occurring.
    /// R16 arenas are excluded since they are a raw K+Q capture format, not compression.
    pub fn count_quantized_arenas(&self) -> Result<(usize, usize)> {
        self.inner.storage.read(|s| {
            let arenas = s.arenas();
            let total = arenas.len();
            let quantized = arenas
                .values()
                .filter(|a| match a {
                    super::Arena::Quantized { format, .. } => {
                        *format != crate::kv_cache::QuantFormat::R16
                    }
                    _ => false,
                })
                .count();
            (quantized, total)
        })
    }

    /// Calculate the percentage of a sequence's tokens that are stored in quantized arenas.
    ///
    /// Returns (quantized_tokens, total_tokens) based on which chunks are in quantized arenas.
    /// Each chunk holds `chunk_size` tokens.
    ///
    /// This validates that the actual ChunkRefs for a sequence point to quantized storage,
    /// not just that quantized arenas exist.
    pub fn quantized_token_stats(&self, batch_idx: usize) -> Result<(usize, usize)> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return Ok((0, 0)), // No sequence at this slot
        };

        // Check all unique arenas referenced by each block's GID slots.
        // A block is counted as "quantized" if ANY of its arenas is quantized.
        self.inner.storage.read(|s| {
            let arenas = s.arenas();
            let mut quantized_tokens = 0usize;
            let total_tokens = slot.block_count() * CHUNK_SIZE;

            for cw in slot.chunks_slice() {
                // A block is "quantized" only if it has a non-R16 quantized arena.
                // R16 is a raw storage format (K+Q capture), not a compression format.
                let any_quant =
                    cw.gids
                        .unique_arena_indices()
                        .iter()
                        .any(|&ai| match arenas.get(&ai) {
                            Some(super::Arena::Quantized { format, .. }) => {
                                *format != crate::kv_cache::QuantFormat::R16
                            }
                            _ => false,
                        });
                if any_quant {
                    quantized_tokens += CHUNK_SIZE;
                }
            }

            (quantized_tokens, total_tokens)
        })
    }

    /// Compute byte-level compression statistics for a sequence.
    pub fn compression_bpe(&self, batch_idx: usize) -> candle::Result<(f64, usize)> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return Ok((0.0f64, 0)),
        };

        if slot.is_empty() {
            return Ok((0.0f64, 0));
        }

        self.inner.storage.read(|s| {
            let arenas = s.arenas();
            // Walk every (head, palette, K/V) slot individually.
            // Each slot covers CHUNK_SIZE × (head_dim / N_PALETTE) elements.
            // Float and R16 slots are excluded from both numerator and denominator
            // so the ratio is CR = 16 / effective_bpe for compressed-only slots.
            let mut actual = 0f64;
            let mut n_quant = 0usize;

            for cw in slot.chunks_slice() {
                for gid in cw.gids.as_slice() {
                    let ai = gid.arena_idx();
                    match arenas.get(&ai) {
                        Some(super::Arena::Float { .. }) => {
                            continue;
                        }
                        Some(super::Arena::Quantized { format, .. })
                            if *format == crate::kv_cache::QuantFormat::R16 =>
                        {
                            continue;
                        }
                        Some(super::Arena::Quantized { format, .. }) => {
                            actual += format.bits_per_elem() as f64 * cw.usage as f64;
                            n_quant += cw.usage as usize;
                        }
                        None => continue,
                    }
                }
            }
            (actual as f64, n_quant)
        })
    }

    pub fn compression_dist(
        &self,
        batch_idx: usize,
        is_value: bool,
        ret: &mut HashMap<GgmlDType, usize>,
    ) {
        let state = self.state.read().unwrap();
        let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return,
        };

        self.inner
            .storage
            .read(|s| {
                let arenas = s.arenas();
                // gids are interleaved: [K_head0, V_head0, K_head1, V_head1, ...]
                // stride by 2 starting at 0 (K) or 1 (V) to avoid counting both sides.
                let start = usize::from(is_value);
                for cw in slot.chunks_slice() {
                    let all = cw.gids.as_slice();
                    let mut i = start;
                    while i < all.len() {
                        let ai = all[i].arena_idx();
                        let dtype = match arenas.get(&ai) {
                            Some(Arena::Float { dtype, .. }) => dtype.to_ggml_dtype(),
                            Some(Arena::Quantized { format, .. }) => format.to_ggml_dtype(),
                            None => {
                                i += 2;
                                continue;
                            }
                        };
                        // Count palette sub-band slots: one per GID (matches per-block diagnostics
                        // where each block shows N_PALETTE × n_kv_head slot counts per format).
                        *(ret.entry(dtype).or_default()) += 1;
                        i += 2;
                    }
                }
            })
            .unwrap();
    }

    /// Dump float K and V data for all chunks in a sequence.
    ///
    /// For each chunk in the sequence at `batch_idx` that resides in a float arena,
    /// returns `(block_idx, k_flat_f32, v_flat_f32)` where each `Vec<f32>` contains
    /// `n_kv_head * chunk_size * head_dim` values in `[head, token, dim]` order.
    ///
    /// Chunks already migrated to quantized arenas are skipped.  For pure float
    /// sessions (F16 / F32 / BF16) all chunks remain float and nothing is skipped.
    pub fn dump_sequence_float_chunks(
        &self,
        batch_idx: usize,
    ) -> Result<Vec<(usize, Vec<f32>, Vec<f32>)>> {
        let n_kv_head = self.inner.n_kv_head;

        // Collect per-block GID snapshots while holding the state lock briefly.
        // Each entry is one (head, palette) sub-band:
        // (k_arena_idx, k_chunk_idx, v_arena_idx, v_chunk_idx).
        //
        // IMPORTANT: `k_gid(h)` / `v_gid(h)` are backward-compat palette-0 aliases.
        // For palette4 layouts we must dump ALL palette sub-bands to reconstruct the
        // full logical head_dim, otherwise the binary dump only contains 1/4 of the
        // head and the offline selection analysis becomes invalid.
        let block_gids: Vec<Vec<(usize, usize, usize, usize)>> = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                Some(s) => s,
                None => return Ok(vec![]),
            };
            slot.chunks_slice()
                .iter()
                .map(|cw| {
                    (0..n_kv_head)
                        .flat_map(|h| {
                            (0..crate::kv_cache::arena_table::N_PALETTE).map(move |p| {
                                let kg = cw.gids.k_gid_pal(h, p);
                                let vg = cw.gids.v_gid_pal(h, p);
                                (
                                    kg.arena_idx(),
                                    kg.chunk_idx(),
                                    vg.arena_idx(),
                                    vg.chunk_idx(),
                                )
                            })
                        })
                        .collect()
                })
                .collect()
        };
        if block_gids.is_empty() {
            return Ok(vec![]);
        }

        let mut result = Vec::with_capacity(block_gids.len());
        for (block_idx, head_gids) in block_gids.iter().enumerate() {
            let maybe_kv: Option<(Vec<f32>, Vec<f32>)> = self.inner.storage.read(|s| {
                let arenas = s.arenas();
                let mut k_all = Vec::with_capacity(n_kv_head * CHUNK_SIZE * self.inner.head_dim);
                let mut v_all = Vec::with_capacity(n_kv_head * CHUNK_SIZE * self.inner.head_dim);

                for &(k_ai, k_ci, v_ai, v_ci) in head_gids.iter() {
                    // Each flat arena chunk stores one head × one palette sub-band × one side.
                    // Concatenating palette order p=0..N_PALETTE reconstructs the full logical head.
                    let k_arena = arenas.get(&k_ai)?;
                    match k_arena {
                        super::Arena::Float { data, .. } => {
                            let head_data = data
                                .narrow(0, k_ci, 1)
                                .ok()?
                                .squeeze(0)
                                .ok()?
                                .to_dtype(DType::F32)
                                .ok()?
                                .flatten_all()
                                .ok()?
                                .to_vec1::<f32>()
                                .ok()?;
                            k_all.extend(head_data);
                        }
                        super::Arena::Quantized { .. } => return None,
                    }

                    let v_arena = arenas.get(&v_ai)?;
                    match v_arena {
                        super::Arena::Float { data, .. } => {
                            let head_data = data
                                .narrow(0, v_ci, 1)
                                .ok()?
                                .squeeze(0)
                                .ok()?
                                .to_dtype(DType::F32)
                                .ok()?
                                .flatten_all()
                                .ok()?
                                .to_vec1::<f32>()
                                .ok()?;
                            v_all.extend(head_data);
                        }
                        super::Arena::Quantized { .. } => return None,
                    }
                }

                Some((k_all, v_all))
            })?;

            if let Some((k_data, v_data)) = maybe_kv {
                result.push((block_idx, k_data, v_data));
            }
        }

        Ok(result)
    }

    /// Dump K (with captured Q) and V data for chunks stored as R16.
    ///
    /// Returns `(block_idx, k_flat_f32, v_flat_f32, q_flat_f32)` for each chunk
    /// whose K side resides in an R16 quantized arena.  Each `Vec<f32>` contains
    /// `n_kv_head * chunk_size * head_dim` values in palette-major layout, matching
    /// `dump_sequence_float_chunks`: `[head][palette][token][sub_dim]`.
    ///
    /// R16 block layout (per (head, palette) sub-arena chunk):
    ///   - sub_head_dim = head_dim / N_PALETTE blocks of 128 bytes each
    ///   - block[d] = { F16 d[32]  // K values for tokens 0..32 at dim d
    ///                , u16 q[32]  // F16 Q values for tokens 0..32 at dim d
    ///                }
    ///
    /// Chunks where K is not R16 (e.g., a partial unsealed tail in float, or
    /// already-migrated quantized chunks) are skipped.
    pub fn dump_sequence_r16_kv_chunks(
        &self,
        batch_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> Result<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> {
        use crate::kv_cache::arena_table::N_PALETTE;
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
        let sub_head_dim = (head_dim / N_PALETTE).max(1);
        // Bytes occupied by a single (head, palette) chunk's R16 storage:
        // sub_head_dim blocks × 128 bytes/block.
        let r16_bytes_per_chunk = sub_head_dim * 128;
        let elems_per_subband = CHUNK_SIZE * sub_head_dim;

        // Resolve (absolute_block_idx, head_gids) for the requested block
        // range.  `block_range = None` walks every chunk in the sequence;
        // `Some((lo, hi))` walks only chunks in `[lo, hi)`.
        let block_gids: Vec<(usize, Vec<(usize, usize, usize, usize)>)> = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                Some(s) => s,
                None => return Ok(vec![]),
            };
            let chunks = slot.chunks_slice();
            let (lo, hi) = match block_range {
                Some((l, h)) => (l.min(chunks.len()), h.min(chunks.len())),
                None => (0, chunks.len()),
            };
            if hi <= lo {
                return Ok(vec![]);
            }
            chunks[lo..hi]
                .iter()
                .enumerate()
                .map(|(i, cw)| {
                    let absolute_idx = lo + i;
                    let gids: Vec<(usize, usize, usize, usize)> = (0..n_kv_head)
                        .flat_map(|h| {
                            (0..N_PALETTE).map(move |p| {
                                let kg = cw.gids.k_gid_pal(h, p);
                                let vg = cw.gids.v_gid_pal(h, p);
                                (
                                    kg.arena_idx(),
                                    kg.chunk_idx(),
                                    vg.arena_idx(),
                                    vg.chunk_idx(),
                                )
                            })
                        })
                        .collect();
                    (absolute_idx, gids)
                })
                .collect()
        };
        if block_gids.is_empty() {
            return Ok(vec![]);
        }

        let total_per_chunk = n_kv_head * N_PALETTE * elems_per_subband;
        let mut result = Vec::with_capacity(block_gids.len());
        for (block_idx, head_gids) in block_gids.iter() {
            // K + Q come from R16 raw bytes per (head, palette) chunk.
            // V comes from Float arenas (R16 mode keeps V float).
            // Layout matches dump_sequence_float_chunks: palette-major within head,
            // each (head, palette) sub-band contributes CHUNK_SIZE × sub_head_dim
            // elements in token-major order.
            let mut k_all: Vec<f32> = Vec::with_capacity(total_per_chunk);
            let mut v_all: Vec<f32> = Vec::with_capacity(total_per_chunk);
            let mut q_all: Vec<f32> = Vec::with_capacity(total_per_chunk);
            let mut ok = true;

            let _ = self.inner.storage.read(|s| {
                let arenas = s.arenas();
                for &(k_ai, k_ci, v_ai, v_ci) in head_gids.iter() {
                    // K + Q from R16 quantized arena.
                    let Some(k_arena) = arenas.get(&k_ai) else {
                        ok = false;
                        return Ok::<(), candle::Error>(());
                    };
                    match k_arena {
                        super::Arena::Quantized { data, format, .. }
                            if *format == crate::kv_cache::QuantFormat::R16 =>
                        {
                            let chunk_off = k_ci * r16_bytes_per_chunk;
                            if chunk_off + r16_bytes_per_chunk > data.storage_size_in_bytes() {
                                ok = false;
                                return Ok(());
                            }
                            // Ranged DtoH: copy ONLY this chunk's bytes from
                            // VRAM, not the whole arena.
                            let chunk_owned =
                                data.data_range(chunk_off..chunk_off + r16_bytes_per_chunk)?;
                            let chunk_bytes: &[u8] = &chunk_owned;
                            // Token-major append into the (head, palette) sub-band.
                            // R16 storage is dim-major: block[d][t]. Reorder to [t][d].
                            for t in 0..CHUNK_SIZE {
                                for d in 0..sub_head_dim {
                                    let blk_off = d * 128;
                                    let k_lo = chunk_bytes[blk_off + t * 2];
                                    let k_hi = chunk_bytes[blk_off + t * 2 + 1];
                                    let k_h = half::f16::from_le_bytes([k_lo, k_hi]);
                                    let q_lo = chunk_bytes[blk_off + 64 + t * 2];
                                    let q_hi = chunk_bytes[blk_off + 64 + t * 2 + 1];
                                    let q_h = half::f16::from_le_bytes([q_lo, q_hi]);
                                    k_all.push(k_h.to_f32());
                                    q_all.push(q_h.to_f32());
                                }
                            }
                        }
                        _ => {
                            ok = false;
                            return Ok(());
                        }
                    }

                    // V from float arena (always float in R16 mode).
                    let Some(v_arena) = arenas.get(&v_ai) else {
                        ok = false;
                        return Ok(());
                    };
                    match v_arena {
                        super::Arena::Float { data, .. } => {
                            let head_data = data
                                .narrow(0, v_ci, 1)?
                                .squeeze(0)?
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?;
                            v_all.extend(head_data);
                        }
                        _ => {
                            ok = false;
                            return Ok(());
                        }
                    }
                }
                Ok(())
            })?;

            if ok {
                result.push((*block_idx, k_all, v_all, q_all));
            }
        }

        Ok(result)
    }

    /// Fast CUDA path: gather R16 K/Q and float-F16 V chunks into a single kernel launch.
    ///
    /// Returns the same `(block_idx, k_flat_f32, v_flat_f32, q_flat_f32)` format as
    /// `dump_sequence_r16_kv_chunks`, but replaces O(n_head × N_PALETTE × n_blocks)
    /// synchronous `memcpy_dtov` calls with:
    ///   1. One tiny HtoD upload of resolved chunk pointers (~8 bytes × n_warps)
    ///   2. One async kernel launch
    ///   3. One synchronous DtoH copy of all three output tensors
    ///
    /// Falls back to `dump_sequence_r16_kv_chunks` on non-CUDA devices.
    /// Skips blocks where K is not R16 or V is not float-F16.
    #[cfg(feature = "cuda")]
    pub fn gather_r16_kv_probe(
        &self,
        batch_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> Result<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> {
        use crate::kv_cache::arena_table::{ArenaFormatTag, N_PALETTE};
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::kernels;

        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
        let sub_head_dim = (head_dim / N_PALETTE).max(1);
        let elems_per_subband = CHUNK_SIZE * sub_head_dim;
        let elems_per_block = n_kv_head * N_PALETTE * elems_per_subband;

        let Device::Cuda(cuda_dev) = &self.inner.device else {
            return self.dump_sequence_r16_kv_chunks(batch_idx, block_range);
        };

        // Step 1: collect (absolute_block_idx, [(k_ai, k_ci, v_ai, v_ci)]) for the range.
        let block_gids: Vec<(usize, Vec<(usize, usize, usize, usize)>)> = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                Some(s) => s,
                None => return Ok(vec![]),
            };
            let chunks = slot.chunks_slice();
            let (lo, hi) = match block_range {
                Some((l, h)) => (l.min(chunks.len()), h.min(chunks.len())),
                None => (0, chunks.len()),
            };
            if hi <= lo {
                return Ok(vec![]);
            }
            chunks[lo..hi]
                .iter()
                .enumerate()
                .map(|(i, cw)| {
                    let absolute_idx = lo + i;
                    let gids: Vec<(usize, usize, usize, usize)> = (0..n_kv_head)
                        .flat_map(|h| {
                            (0..N_PALETTE).map(move |p| {
                                let kg = cw.gids.k_gid_pal(h, p);
                                let vg = cw.gids.v_gid_pal(h, p);
                                (
                                    kg.arena_idx(),
                                    kg.chunk_idx(),
                                    vg.arena_idx(),
                                    vg.chunk_idx(),
                                )
                            })
                        })
                        .collect();
                    (absolute_idx, gids)
                })
                .collect()
        };
        if block_gids.is_empty() {
            return Ok(vec![]);
        }

        // Step 2: resolve arena info and build per-warp pointer arrays.
        // Only include blocks where all (h,p) K sides are R16 and V sides are F16.
        let arena_info = self.inner.resolve_arena_info()?;

        let mut k_ptrs: Vec<i64> = Vec::new();
        let mut v_ptrs: Vec<i64> = Vec::new();
        let mut r16_block_indices: Vec<usize> = Vec::new();

        for (block_idx, head_gids) in &block_gids {
            let is_r16 = head_gids.iter().all(|&(k_ai, _, _, _)| {
                arena_info
                    .get(k_ai)
                    .map_or(false, |a| a.k_format_tag == ArenaFormatTag::R16)
            });
            if !is_r16 {
                continue;
            }
            let is_v_f16 = head_gids.iter().all(|&(_, _, v_ai, _)| {
                arena_info
                    .get(v_ai)
                    .map_or(false, |a| a.v_format_tag == ArenaFormatTag::F16)
            });
            if !is_v_f16 {
                continue;
            }

            // Guard: both K (R16) and V (Float F16) arenas must be GPU-backed.
            // CPU-backed arenas have base_ptr=0; passing 0+ci*stride to the kernel
            // produces a small non-zero address that CUDA maps to unallocated memory
            // → ILLEGAL_ADDRESS fault on the GPU stream.
            let all_ptrs_nonzero = head_gids.iter().all(|&(k_ai, _, v_ai, _)| {
                arena_info.get(k_ai).map_or(false, |a| a.base_ptr != 0)
                    && arena_info.get(v_ai).map_or(false, |a| a.base_ptr != 0)
            });
            if !all_ptrs_nonzero {
                tracing::error!(
                    block_idx,
                    "gather_r16_kv_probe: null arena base_ptr — arena not yet \
                     GPU-resident, skipping block"
                );
                continue;
            }

            r16_block_indices.push(*block_idx);
            for &(k_ai, k_ci, v_ai, v_ci) in head_gids.iter() {
                let k_arena = &arena_info[k_ai];
                let v_arena = &arena_info[v_ai];
                k_ptrs.push(k_arena.base_ptr as i64 + k_ci as i64 * k_arena.chunk_byte_stride);
                v_ptrs.push(v_arena.base_ptr as i64 + v_ci as i64 * v_arena.chunk_byte_stride);
            }
        }
        let mut result: Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)> =
            Vec::with_capacity(block_gids.len());

        if !r16_block_indices.is_empty() {
            let n_r16_blocks = r16_block_indices.len();
            let n_warps = n_r16_blocks * n_kv_head * N_PALETTE;
            let total_elems = n_warps * elems_per_subband;

            // Step 3: HtoD upload of pointer arrays, alloc combined output, launch.
            let k_ptrs_gpu = cuda_dev.memcpy_stod(&k_ptrs)?;
            let v_ptrs_gpu = cuda_dev.memcpy_stod(&v_ptrs)?;

            // Combined buffer: [K section][Q section][V section], each `total_elems` halves.
            let out_kqv = unsafe { cuda_dev.alloc::<half::f16>(3 * total_elems)? };

            let stream = cuda_dev.cuda_stream();
            {
                let (kp, _kg) = k_ptrs_gpu.device_ptr(&stream);
                let (vp, _vg) = v_ptrs_gpu.device_ptr(&stream);
                let (okqv, _g) = out_kqv.device_ptr(&stream);
                candle::set_kernel_breadcrumb("run_gather_r16_kv_f16", file!(), line!());
                unsafe {
                    kernels::simple::gather_r16_kv::run_gather_r16_kv_f16(
                        kp as *const i64,
                        vp as *const i64,
                        okqv as *mut std::ffi::c_void,
                        n_warps as i32,
                        sub_head_dim as i32,
                        stream.cu_stream() as *mut _,
                    );
                }
            }

            // Step 4: single DtoH copy of combined K/Q/V output.
            let kqv_cpu: Vec<half::f16> = cuda_dev.memcpy_dtov(&out_kqv)?;
            let k_cpu = &kqv_cpu[..total_elems];
            let q_cpu = &kqv_cpu[total_elems..2 * total_elems];
            let v_cpu = &kqv_cpu[2 * total_elems..];

            // Step 5: transpose d-major kernel output → token-major and convert F16 → F32.
            //
            // The kernel writes d-major within each warp (= one head×palette sub-band):
            //   kqv[warp_off + d * CHUNK_SIZE + token]
            // The consumer (r16_block_to_turn_signatures) expects token-major:
            //   q_flat[warp_off + token * sub_head_dim + d]
            //
            // We fold the transpose into the F16→F32 conversion — no extra allocation.
            let n_subbands = n_kv_head * N_PALETTE;
            for (bi, block_idx) in r16_block_indices.iter().enumerate() {
                let mut k_f32 = Vec::with_capacity(elems_per_block);
                let mut v_f32 = Vec::with_capacity(elems_per_block);
                let mut q_f32 = Vec::with_capacity(elems_per_block);
                for warp_local in 0..n_subbands {
                    let warp_off = bi * elems_per_block + warp_local * elems_per_subband;
                    for t in 0..CHUNK_SIZE {
                        for d in 0..sub_head_dim {
                            // d-major read → token-major push order
                            let src = warp_off + d * CHUNK_SIZE + t;
                            k_f32.push(k_cpu[src].to_f32());
                            v_f32.push(v_cpu[src].to_f32());
                            q_f32.push(q_cpu[src].to_f32());
                        }
                    }
                }
                result.push((*block_idx, k_f32, v_f32, q_f32));
            }
        }

        Ok(result)
    }

    /// Get K arenas as float tensors.
    ///
    /// With flat arenas, each chunk stores one head's one side.
    /// K data is addressed by chunk index via HeadGids — no narrowing needed.
    /// Returns the full arena tensor (each chunk is a single head's single side).
    pub fn k_arenas(&self) -> Vec<Tensor> {
        self.inner
            .storage
            .read(|s| {
                s.arenas()
                    .values()
                    .filter_map(|a| a.as_float_data().cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get V arenas as float tensors.
    ///
    /// With flat arenas, K and V share the same arena format (each chunk is
    /// one head's one side).  V data is addressed by chunk index via HeadGids.
    pub fn v_arenas(&self) -> Vec<Tensor> {
        self.inner
            .storage
            .read(|s| {
                s.arenas()
                    .values()
                    .filter_map(|a| a.as_float_data().cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get float arenas (K, V). Returns None if default format is quantized.
    /// With flat arenas, K and V are the same tensor — chunks are distinguished
    /// by GID, not by position within the arena.
    pub fn float_arenas(&self) -> Option<(Vec<Tensor>, Vec<Tensor>)> {
        if self.inner.storage.is_quantized() {
            return None;
        }
        self.inner
            .storage
            .read(|s| {
                let all: Vec<_> = s
                    .arenas()
                    .values()
                    .filter_map(|a| a.as_float_data().cloned())
                    .collect();
                (all.clone(), all)
            })
            .ok()
    }

    /// Get quantized arenas (K, V). Returns None if default format is float.
    /// Note: With heterogeneous storage, some arenas may be float even if this returns Some.
    pub fn quantized_arenas(&self) -> Option<(Vec<QTensor>, Vec<QTensor>)> {
        if !self.inner.storage.is_quantized() {
            return None;
        }
        self.inner
            .storage
            .read(|s| {
                let k: Vec<_> = s
                    .arenas()
                    .values()
                    .filter_map(|a| a.as_quantized_data().cloned())
                    .collect();
                let v: Vec<_> = k.clone();
                (k, v)
            })
            .ok()
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

    /// Build the combined chunk_meta row for one batch slot.
    ///
    /// Returns one [`ChunkMeta`] entry per logical block.  Pass the result to
    /// [`ChunkMeta::into_u32s`] to flatten it for [`candle::Tensor::from_vec`].
    ///
    /// `seq_len` is the current total token count for the sequence; it is used
    /// to compute the active tail usage dynamically.
    pub fn chunk_meta_row(&self, batch_idx: usize, seq_len: usize) -> Vec<ChunkMeta> {
        let state = self.state.read().expect("chunked state lock poisoned");
        let max_blocks = state.max_blocks;
        let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return vec![ChunkMeta::default(); max_blocks],
        };

        // Build block entries from chunks.
        // For the last block, compute usage dynamically from seq_len so that the
        // chunk_meta correctly reflects tokens about to be written (decode) or
        // being written in the current prefill batch.  This mirrors the old
        // active-block dynamic-usage path in HEAD.
        let n = slot.block_count();
        let mut committed: usize = 0;
        let mut row: Vec<ChunkMeta> = slot
            .chunks_slice()
            .iter()
            .enumerate()
            .map(|(i, cw)| {
                let rope = committed as i32;
                let usage = if i + 1 == n && seq_len > 0 {
                    // Last (write-target) block: derive from seq_len so that tokens
                    // not yet committed via set_current_seq_len are still visible.
                    (seq_len.saturating_sub(committed)).min(CHUNK_SIZE) as u32
                } else {
                    cw.usage
                };
                committed += usage as usize;
                ChunkMeta::new(usage, rope, cw.offset)
            })
            .collect();

        // Pad with default (unmapped) entries to reach max_blocks.
        row.resize(max_blocks, ChunkMeta::default());
        row
    }

    /// Get the max_blocks for this backing (width of the block table).
    pub fn max_blocks(&self) -> usize {
        self.state
            .read()
            .expect("chunked state lock poisoned")
            .max_blocks
    }

    /// Report GPU memory usage broken down by arena format.
    ///
    /// Returns a Vec of `(format_label, arena_count, total_bytes)` tuples,
    /// one entry per unique format (e.g. "BF16", "F16", "Q8_0").
    pub fn memory_report(&self) -> Result<Vec<(String, usize, usize)>> {
        self.inner.storage.read(|s| {
            let mut by_format: std::collections::BTreeMap<String, (usize, usize)> =
                std::collections::BTreeMap::new();
            for arena in s.arenas().values() {
                let label = arena.format_label();
                let bytes = arena.gpu_memory_bytes();
                let entry = by_format.entry(label).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += bytes;
            }
            by_format
                .into_iter()
                .map(|(label, (count, bytes))| (label, count, bytes))
                .collect()
        })
    }

    /// Total GPU memory used by all arenas in this backing (bytes).
    pub fn total_arena_gpu_bytes(&self) -> Result<usize> {
        self.inner
            .storage
            .read(|s| s.arenas().values().map(|a| a.gpu_memory_bytes()).sum())
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

/// Collect a combined memory report from ALL registered ChunkedKvBacking instances.
///
/// Returns a Vec of `(backing_index, format_label, arena_count, total_bytes)`.
/// Dead (dropped) backings are skipped.
pub fn global_arena_memory_report() -> Vec<(usize, String, usize, usize)> {
    let mut results = Vec::new();
    if let Ok(registry) = BACKING_REGISTRY.lock() {
        for (layer_idx, weak) in registry.iter().enumerate() {
            if let Some(backing) = weak.upgrade() {
                let backing = ChunkedKvBacking {
                    inner: backing,
                    layer_idx,
                    state: std::sync::Arc::new(std::sync::RwLock::new(
                        super::types::BlockTableState::new(layer_idx, 1, 0),
                    )),
                };
                if let Ok(report) = backing.memory_report() {
                    for (label, count, bytes) in report {
                        results.push((layer_idx, label, count, bytes));
                    }
                }
            }
        }
    }
    results
}

/// Get the total GPU memory used by all registered arena backings (bytes).
pub fn global_arena_gpu_bytes() -> usize {
    let mut total = 0;
    if let Ok(registry) = BACKING_REGISTRY.lock() {
        for (layer_idx, weak) in registry.iter().enumerate() {
            if let Some(backing) = weak.upgrade() {
                let backing = ChunkedKvBacking {
                    inner: backing,
                    layer_idx,
                    state: std::sync::Arc::new(std::sync::RwLock::new(
                        super::types::BlockTableState::new(layer_idx, 1, 0),
                    )),
                };
                if let Ok(bytes) = backing.total_arena_gpu_bytes() {
                    total += bytes;
                }
            }
        }
    }
    total
}

/// Print a detailed per-arena diagnostic table for ALL registered backings to stderr.
///
/// Aggregates arena rows across all live backings into a single table.  Each row
/// is tagged with its backing index (registry slot) — NOT a layer index, since
/// arenas are typically shared across all layers.  Empty backings are omitted.
pub fn global_print_arena_table() {
    use crate::kv_cache::chunked::arena::ArenaRow;
    use std::collections::BTreeMap;

    // Collect rows from all live backings, tagging each with its backing index.
    let mut tagged_rows: Vec<(usize, ArenaRow)> = Vec::new();
    let mut live_backings = 0usize;
    let mut dead_backings = 0usize;

    if let Ok(registry) = BACKING_REGISTRY.lock() {
        for (layer_idx, weak) in registry.iter().enumerate() {
            if let Some(inner) = weak.upgrade() {
                live_backings += 1;
                let backing = ChunkedKvBacking {
                    inner,
                    layer_idx,
                    state: std::sync::Arc::new(std::sync::RwLock::new(
                        super::types::BlockTableState::new(layer_idx, 1, 0),
                    )),
                };
                if let Ok(rows) = backing.arena_rows() {
                    if !rows.is_empty() {
                        for row in rows {
                            tagged_rows.push((layer_idx, row));
                        }
                    }
                }
            } else {
                dead_backings += 1;
            }
        }
    }

    if live_backings == 0 {
        eprintln!(
            "\n[arena-table] WARNING: no live backings in registry — \
             all ChunkedKvBacking instances were already dropped at diagnostic capture time. \
             Arena data unavailable."
        );
        return;
    }

    // ── Fixed-width box ──
    // W is the number of characters between the two ║ bookends.
    // Every content line is right-padded to exactly W chars.
    const W: usize = 92;
    let sep = "═".repeat(W);
    let thin = "─".repeat(W);

    // Helper: right-pad a string to W, wrap with ║ bookends, print to stderr.
    let row = |s: &str| {
        if s.len() >= W {
            eprintln!("║{}║", &s[..W]);
        } else {
            eprintln!("║{}{:pad$}║", s, "", pad = W - s.len());
        }
    };

    let backings_with_arenas = tagged_rows
        .iter()
        .map(|(b, _)| *b)
        .collect::<std::collections::BTreeSet<_>>()
        .len();

    eprintln!("\n╔{}╗", sep);
    row(&format!(
        "{:^width$}",
        "ARENA TABLE (all backings)",
        width = W
    ));
    row(&format!(
        "{:^width$}",
        format!(
            "{} live backings, {} dead, {} with arenas",
            live_backings, dead_backings, backings_with_arenas
        ),
        width = W
    ));
    eprintln!("╠{}╣", sep);

    // Column layout:  Bck(4) 2 Idx(4) 2 Type(22) 2 Cap(6) 1 HWM(5) 1 Active(6) 1 Free(5) 1 GPU_MiB(10) 1 Flags(11) = 90 + 2 bookends = 92
    row(&format!(
        " {:>4}  {:>4}  {:<22}  {:>6} {:>5} {:>6} {:>5} {:>10} {:<11}",
        "Bck", "Idx", "Type", "Cap", "HWM", "Active", "Free", "GPU MiB", "Flags"
    ));
    eprintln!("║{}║", thin);

    if tagged_rows.is_empty() {
        row(&format!("{:^width$}", "(no arenas allocated)", width = W));
    } else {
        for (backing_idx, r) in &tagged_rows {
            let mib = format!("{:.1}", r.gpu_bytes as f64 / (1024.0 * 1024.0));
            let flags = if r.is_tombstone {
                "TOMBSTONE"
            } else if r.is_full() {
                "FULL"
            } else if r.is_empty() {
                "empty"
            } else {
                ""
            };
            row(&format!(
                " {:>4}  {:>4}  {:<22}  {:>6} {:>5} {:>6} {:>5} {:>10} {:<11}",
                backing_idx,
                r.arena_idx,
                r.type_label,
                r.capacity,
                r.hwm,
                r.active,
                r.free_list,
                mib,
                flags
            ));
        }
    }

    // ── Summary by type (aggregated across all backings) ──
    eprintln!("╠{}╣", sep);
    row(&format!("{:^width$}", "Summary by type", width = W));
    eprintln!("║{}║", thin);

    let mut by_type: BTreeMap<String, (usize, usize, usize, usize, usize)> = BTreeMap::new();
    let mut tombstone_count = 0usize;
    let mut full_count = 0usize;
    let mut empty_count = 0usize;

    for (_, r) in &tagged_rows {
        if r.is_tombstone {
            tombstone_count += 1;
            continue;
        }
        let e = by_type
            .entry(r.type_label.clone())
            .or_insert((0, 0, 0, 0, 0));
        e.0 += 1; // arena count
        e.1 += r.active;
        e.2 += r.capacity;
        e.3 += r.gpu_bytes;
        e.4 += r.free_list;
        if r.is_full() {
            full_count += 1;
        }
        if r.is_empty() {
            empty_count += 1;
        }
    }

    for (label, (count, active, capacity, bytes, frag)) in &by_type {
        let mib = *bytes as f64 / (1024.0 * 1024.0);
        let frag_pct = if *capacity > 0 {
            (*frag as f64 / *capacity as f64) * 100.0
        } else {
            0.0
        };
        row(&format!(
            "  {:<20} arenas={:<4} used={}/{:<6} free_list={:<4} ({:>4.1}%) {:>8.1} MiB",
            label, count, active, capacity, frag, frag_pct, mib
        ));
    }

    let total_gpu_mib: f64 =
        tagged_rows.iter().map(|(_, r)| r.gpu_bytes).sum::<usize>() as f64 / (1024.0 * 1024.0);
    eprintln!("║{}║", thin);
    row(&format!(
        "  Tombstoned: {:<4}  Full: {:<4}  Empty: {:<4}  Total GPU: {:>10.1} MiB",
        tombstone_count, full_count, empty_count, total_gpu_mib
    ));
    eprintln!("╚{}╝", sep);
}

// ==================== PagedKvArenas Trait Implementation ====================

impl crate::kv_cache::PagedKvArenas for ChunkedKvBacking {
    fn n_kv_head(&self) -> usize {
        self.inner.n_kv_head
    }

    fn head_dim(&self) -> usize {
        self.inner.head_dim
    }

    fn k_format(&self) -> KvFormat {
        ChunkedKvBacking::k_format(self)
    }

    fn v_format(&self) -> KvFormat {
        ChunkedKvBacking::v_format(self)
    }

    fn float_arenas(&self) -> Option<(Vec<Tensor>, Vec<Tensor>)> {
        ChunkedKvBacking::float_arenas(self)
    }

    fn quantized_arenas(&self) -> Option<(Vec<QTensor>, Vec<QTensor>)> {
        ChunkedKvBacking::quantized_arenas(self)
    }
}
