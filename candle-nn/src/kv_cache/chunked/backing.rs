//! BackingInner and ChunkedKvBacking implementation.
//!
//! This module contains:
//! - `BackingInner` - The inner shared state
//! - `ChunkedKvBacking` - The main public API with constructors and accessors

use ahash::{HashMap, HashMapExt};
use candle::quantized::GgmlDType;
use std::cmp;
use std::sync::{Arc, Mutex, RwLock, Weak};

use candle::quantized::pinned_staging::{Generation, PinnedStager};
// Core tensor types, needed on every build — the CPU one included. (These were
// gated on `cuda`, which is why a non-cuda build could not resolve `Device`,
// `DType`, or the crate's own `Result` anywhere in this file.)
use candle::{DType, Device, Result};

use super::head_gids::ChunkBands;
use super::{
    Arena, ArenaStorage, ArenaStorageState, BlockTableState, ChunkMeta, CompressionPolicy,
    GpuArenaClassStats, LiveChunkRef, SealedChunk, StoragePolicy,
};
// Only the CUDA-gated compress-eligibility helper needs the sealed-sequence type.
use super::size_class::{class_for_payload, payload_bytes_for_tag, SizeClass};
#[cfg(feature = "cuda")]
use super::SealedSequence;
use crate::kv_cache::arena_table::{ArenaFormatTag, ArenaLocation, PerHeadEntry};
// `N_PALETTE` is referenced by the intra-doc links throughout this file and by
// the CUDA table builders; without `cuda` only the doc links are left, and they
// still need it in scope to resolve.
#[cfg_attr(not(feature = "cuda"), allow(unused_imports))]
use crate::kv_cache::{HeadGids, KvFormat, QuantFormat, ResolvedArenaInfo, N_PALETTE};
use crate::{CHUNK_SIZE, GID_STRIDE};

/// Substring embedded in the error returned when a GPU KV arena cannot be
/// allocated within the VRAM budget (even after a forced compaction). The
/// scheduler matches on this via [`is_device_oom`] to drive eviction / batch
/// clipping instead of letting the driver spill KV to host memory (which on
/// WDDM silently collapses throughput rather than failing).
pub const KV_DEVICE_OOM_MARKER: &str = "kv-cache GPU VRAM budget exceeded";

/// Recognize a device-out-of-memory error — either our own budget rejection
/// ([`KV_DEVICE_OOM_MARKER`]) or a driver-reported CUDA OOM (when sysmem
/// fallback is disabled, `cuMemAlloc` returns `CUDA_ERROR_OUT_OF_MEMORY`).
pub fn is_device_oom(err: &candle::Error) -> bool {
    let s = err.to_string();
    s.contains(KV_DEVICE_OOM_MARKER)
        || s.contains("out of memory")
        || s.contains("OUT_OF_MEMORY")
        || s.contains("CUDA_ERROR_OUT_OF_MEMORY")
}

/// Global registry of all ChunkedKvBacking instances for cooperative compaction.
/// When one backing needs to grow arenas but fails (OOM), it can ask others to compact.
static BACKING_REGISTRY: Mutex<Vec<Weak<BackingInner>>> = Mutex::new(Vec::new());

pub(super) fn register_backing(inner: &Arc<BackingInner>) {
    if let Ok(mut registry) = BACKING_REGISTRY.lock() {
        // Clean up dead entries while we're here
        registry.retain(|w| w.strong_count() > 0);
        registry.push(Arc::downgrade(inner));
    }
}

/// Decode one band's slot into `f32`, for the offline dump paths.
///
/// `None` unless the band's tag names a **float** format: these dumps
/// reconstruct raw activations, and a quantized band would need the
/// dequantize path rather than a widening cast. The tag is the only record of
/// how to read the slot — the arena is a run of untyped bytes.
///
/// The decode is done on the host from the slot's raw bytes rather than by
/// viewing the slab as a typed tensor, so it works identically for a hot GPU
/// arena and a warm CPU one, and needs no device-side reinterpretation.
fn read_float_band(
    arena: &Arena,
    chunk_idx: usize,
    tag: ArenaFormatTag,
    elems: usize,
) -> Option<Vec<f32>> {
    let dtype = tag.to_dtype()?;
    let bytes = arena
        .slot_bytes(chunk_idx, elems * dtype.size_in_bytes())
        .ok()?
        .to_vec1::<u8>()
        .ok()?;
    let out = match dtype {
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F8E4M3 => bytes
            .iter()
            .map(|b| float8::F8E4M3::from_bits(*b).to_f32())
            .collect(),
        _ => return None,
    };
    Some(out)
}

fn register_state(inner: &Arc<BackingInner>, state: &Arc<RwLock<BlockTableState>>) {
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
    pub(crate) meta_pool: super::meta_pool::MetaPool,

    /// Reused device scratch for the provenance sign-pack kernel — grows to the
    /// largest batch seen and is reused thereafter, so the per-scope seal pays no
    /// device `alloc`/pointer-`memcpy_stod` (the WDDM-expensive part). Held in a
    /// `Mutex` for interior mutability behind the shared `Arc<BackingInner>`; the
    /// seal runs single-threaded so it's uncontended. Mirrors `KvSamplerGpu`.
    #[cfg(feature = "cuda")]
    pub(crate) prov_sign_scratch: Mutex<Option<ProvSignScratch>>,

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
    pub(crate) deferred_arenas: Mutex<Vec<super::arena::ArenaKey>>,
}

/// Grow-only device scratch for [`ChunkedKvBacking::run_prov_sign_pack`].
#[cfg(feature = "cuda")]
pub(crate) struct ProvSignScratch {
    /// Concatenated Q-chunk pointers (capacity ≥ the batch's `n_warps`).
    ptrs: candle::cuda_backend::cudarc::driver::CudaSlice<i64>,
    /// Packed sign bits (capacity ≥ `n_warps × CHUNK_SIZE`).
    out: candle::cuda_backend::cudarc::driver::CudaSlice<u32>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for ProvSignScratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProvSignScratch")
            .field("ptrs_cap", &self.ptrs.len())
            .field("out_cap", &self.out.len())
            .finish()
    }
}

impl BackingInner {
    /// Per-head band count for the arena/record layout: [`LATENT_N_BANDS`] on
    /// the single-latent path, [`N_PALETTE`] for GQA. Every band-count use in
    /// allocation, arena sub-band sizing, and record serialization goes through
    /// this so the single-latent path can carry 8 bands while GQA stays at 4.
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

    pub(super) fn release_empty_arenas(&self) -> Result<usize> {
        let mut freed = 0;

        // Phase 1: Pool-driven tombstoning of fully-free arenas.
        // For each format key, repeatedly call next_tombstone until exhausted.
        let keys = self.pool.format_keys();
        for key in keys {
            while let Some(arena_idx) = self.pool.next_tombstone(key) {
                self.storage.release_arena(arena_idx)?;
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
        // Resize the resident-record stride to match the new band count
        // (single-latent = 8 bands, GQA = 4). Must run before any chunk is
        // allocated — the meta-pool slabs are still empty here — otherwise
        // serialize_kv_heads would overrun the 4-band slots. Idempotent.
        let rb = super::meta_pool::chunk_record_bytes(
            self.inner.n_kv_head,
            self.inner.head_dim,
            self.inner.n_palette(),
        );
        self.inner.meta_pool.set_record_bytes(rb);
        // The constructor's `warm_protected_arenas` ran while single_latent was
        // still false, so it minted the writer/candidate arenas at the GQA band
        // width (`head_dim / N_PALETTE` = 128). The single latent needs them at
        // its own width (`head_dim / 16` = 32) — the width `resolve_arena_info`
        // and the band writes both assume. Nothing has allocated a chunk yet
        // (this runs immediately after construction, before any write), so drop
        // the mis-sized warm arenas; the pool keeps their registrations and the
        // next allocation recreates each at the single-latent width. GQA never
        // calls this, so its warm set is untouched.
        if on {
            let _ = self.inner.storage.truncate_arenas(0);
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
            super::region_pool::ensure(&cuda.cuda_stream())?;
        }

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
            meta_pool: super::meta_pool::MetaPool::new(
                // GQA record stride (4-band); single-latent backings resize this
                // to 8-band via `set_single_latent`, before any chunk allocates.
                super::meta_pool::chunk_record_bytes(
                    n_kv_head,
                    head_dim,
                    crate::kv_cache::arena_table::N_PALETTE,
                ),
                device.clone(),
            ),
            #[cfg(feature = "cuda")]
            prov_sign_scratch: Mutex::new(None),
            single_latent: std::sync::atomic::AtomicBool::new(false),
            mapped_sealed: std::sync::atomic::AtomicBool::new(false),
            deferred_arenas: Mutex::new(Vec::new()),
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

    /// Device address of a resident KV-head record, for embedding into a slice's
    /// `kvheads_ptr`. Returns 0 for a CPU/host-only pool (no device residence).
    /// The address is cached on the handle at allocation, so this is a field read.
    pub fn meta_device_addr(&self, meta: &super::meta_pool::MetaGid) -> u64 {
        meta.device_addr()
    }

    /// Build device-resident KV-head metadata records for a batch of chunks and
    /// return one handle per input. Serializes each `KvHead[n_kv_head]` record
    /// (pal/scale/fmt + the 8 per-palette pointers resolved against `arena_info`
    /// at the chunk's current placement) and uploads them in **one coalesced
    /// transfer** (a single `memcpy_htod` per contiguous slab run) rather than a
    /// tiny copy per chunk. Called at the finalization sites — quantize,
    /// cold-load, and warm→hot elevate — so a resident record always matches the
    /// bytes it describes.
    ///
    /// Returns `None` for every input when the pool has no device residence
    /// (CPU / host-only tier): there is no readable record address, so the
    /// caller must keep `meta = None` and fall back to per-forward scratch heads.
    /// This preserves the invariant `meta.is_some() ⇒ device_addr != 0` that the
    /// prefill/glue serializer relies on (it builds no scratch heads for resident
    /// chunks). Each handle is stored on the `SealedChunk`/`ChunkWindow` and
    /// shared by every slot that references the chunk.
    #[allow(dead_code)] // callers are cuda-gated; a pure-CPU build sees none
    pub(crate) fn build_meta_records(
        &self,
        chunks: &[super::meta_pool::ChunkRecordSrc<'_>],
        arena_info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
    ) -> Result<Vec<Option<super::meta_pool::MetaGid>>> {
        if !self.inner.meta_pool.is_device_resident() {
            return Ok(vec![None; chunks.len()]);
        }
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
        let n_palette = self.inner.n_palette();
        let rb = super::meta_pool::chunk_record_bytes(n_kv_head, head_dim, n_palette);
        let mut items: Vec<(super::meta_pool::MetaGid, Vec<u8>)> = Vec::with_capacity(chunks.len());
        for src in chunks {
            let handle = self.inner.meta_pool.allocate()?;
            let mut bytes = vec![0u8; rb];
            super::meta_pool::serialize_kv_heads(
                &mut bytes, src, n_kv_head, head_dim, n_palette, arena_info,
            );
            items.push((handle, bytes));
        }
        self.inner.meta_pool.write_records_batched(&items)?;
        Ok(items.into_iter().map(|(h, _)| Some(h)).collect())
    }

    /// Release any fully-empty arenas back to the pool **without** the
    /// expensive chunk-moving defrag — the cheap half of compaction. Returns
    /// arenas freed. Used by the scheduler's pressure path so it never spends
    /// seconds on a speculative defrag (which, during active prefills, is
    /// usually futile — the free space sits in protected arenas); the costly
    /// defrag stays on the reactive allocation-time OOM retry.
    pub fn release_empty_arenas(&self) -> Result<usize> {
        self.inner.release_empty_arenas()
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

    /// Elements one chunk slot holds at this backing's geometry — one
    /// `(head, palette-band, side)` of `CHUNK_SIZE` tokens.
    ///
    /// The number every size-class and payload-length question is asked with,
    /// exposed because callers outside this crate (the persistence gather)
    /// need it to turn a band's format tag into a byte length.
    pub fn elems_per_chunk(&self) -> usize {
        self.inner.elems_per_chunk()
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

    /// Per-chunk `(offset, len, cum_before)` real-token window for a sequence — the
    /// exact layout attention reads (writer chunk gets the `seq_offset`-derived
    /// length). A provenance / diagnostic gather consults this to check only real
    /// slots and skip partial-chunk padding. See
    /// [`super::types::SequenceState::provenance_chunk_layout`].
    pub fn provenance_chunk_layout(
        &self,
        batch_idx: usize,
        seq_offset: usize,
    ) -> Vec<(u16, u16, usize)> {
        let Ok(state) = self.state.read() else {
            return Vec::new();
        };
        match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
            Some(seq) => seq.provenance_chunk_layout(seq_offset),
            None => Vec::new(),
        }
    }

    /// Visit `batch_idx`'s live chunks as borrowed [`LiveChunkRef`]s under
    /// the state read lock — the zero-clone path for per-forward metadata
    /// builds, avoiding the per-chunk clones + `arena_byte_size` walks that
    /// dominate host cost at deep prefixes. [`Self::live_chunks_as_sealed`]
    /// serves callers that need OWNED snapshots (seal/persist/migration).
    pub fn visit_live_chunks<R>(
        &self,
        batch_idx: usize,
        f: impl FnOnce(&mut dyn Iterator<Item = LiveChunkRef<'_>>) -> R,
    ) -> Option<R> {
        let state = self.state.read().ok()?;
        let seq = state.sequences.get(batch_idx)?.as_ref()?;
        let mut it = seq.chunks_slice().iter().map(|cw| LiveChunkRef {
            gids: &cw.gids,
            offset: cw.offset,
            token_count: cw.usage as u16,
            k_pal: cw.k_pal.as_slice(),
            v_pal: cw.v_pal.as_slice(),
            k_scale: cw.k_scale.as_slice(),
            v_scale: cw.v_scale.as_slice(),
            k_fmt: cw.k_fmt.as_slice(),
            v_fmt: cw.v_fmt.as_slice(),
            meta: cw.meta.as_ref(),
        });
        Some(f(&mut it))
    }

    /// Drop the sequence's cached GPU slot-state buffer so the next metadata
    /// build re-serialises it from the authoritative CPU chunk state (the
    /// REBUILD path). Needed when host bookkeeping (`set_len` after a
    /// truncate) has changed slice lens/rope bases without touching the
    /// serialized buffer — `set_len` deliberately never writes the pinned DMA
    /// source (see its comment), so a later build's REUSE path would snapshot
    /// stale lens. The speculative-verify wave calls this before building its
    /// virtual-slot headers.
    pub fn invalidate_decode_slot(&self, batch_idx: usize) {
        if let Ok(mut state) = self.state.write() {
            if let Some(Some(seq)) = state.sequences.get_mut(batch_idx) {
                seq.invalidate_gpu_chunks();
            }
        }
    }

    pub fn live_chunks_as_sealed(&self, batch_idx: usize) -> Option<Vec<SealedChunk>> {
        let elems = self.inner.elems_per_chunk();
        let state = self.state.read().ok()?;
        let seq = state.sequences.get(batch_idx)?.as_ref()?;
        let chunks: Vec<SealedChunk> = seq
            .chunks_slice()
            .iter()
            .map(|cw| {
                let byte_size = cw.byte_size(elems);
                SealedChunk {
                    gids: cw.gids.clone(),
                    offset: cw.offset,
                    token_count: cw.usage as u16,
                    k_pal: cw.k_pal.clone(),
                    v_pal: cw.v_pal.clone(),
                    k_scale: cw.k_scale.clone(),
                    v_scale: cw.v_scale.clone(),
                    k_fmt: cw.k_fmt.clone(),
                    v_fmt: cw.v_fmt.clone(),
                    byte_size,
                    // Live snapshot shares the chunk's GIDs — propagate the
                    // resident record handle so the prefill/glue serializer can
                    // emit its `kvheads_ptr` instead of rebuilding heads.
                    meta: cw.meta.clone(),
                }
            })
            .collect();
        Some(chunks)
    }

    /// Patch each sequence's cached decode slot-state WRITER slice after a
    /// mid-decode prefill wrote tokens into the writer chunk in place — see
    /// [`super::types::SequenceState::refresh_decode_writer_slice`]. O(1) per
    /// sequence per layer; sequences with no cached buffer (never decoded, or
    /// cleared by a chunk-boundary append) rebuild fully on the next decode
    /// sync instead.
    /// Free token capacity of `batch_idx`'s current decode WRITE chunk — how
    /// many appended tokens it can still hold before the next append crosses
    /// into a fresh chunk. The caller that extends a slot by `n` tokens uses
    /// this to pick between the O(1) writer-slice patch (`n` fits) and full
    /// invalidation (`n` spans into a new chunk, whose serialized
    /// predecessors would otherwise keep their pre-extension lengths).
    /// `None` when the slot is unallocated or empty.
    pub fn decode_writer_room(&self, batch_idx: usize) -> Option<usize> {
        let state = self.state.read().ok()?;
        let seq = state.sequences.get(batch_idx)?.as_ref()?;
        let chunks = seq.chunks_slice();
        if chunks.is_empty() {
            return None;
        }
        let wi = seq.decode_write_chunk_idx().min(chunks.len() - 1);
        let cw = &chunks[wi];
        Some(CHUNK_SIZE.saturating_sub(cw.offset as usize + cw.usage as usize))
    }

    pub fn refresh_decode_writer_slice(&self, batch_entries: &[(usize, usize)]) -> Result<()> {
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;
        let arena_info = self.resolve_arena_info()?;
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        for &(seq_idx, _) in batch_entries {
            if let Some(Some(seq)) = state.sequences.get_mut(seq_idx) {
                seq.refresh_decode_writer_slice(n_kv_head, head_dim, &arena_info)?;
            }
        }
        Ok(())
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
                                let stride = GID_STRIDE;
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

    /// Like [`Self::sync_decode_gpu_chunks`], but each returned `slices_ptr` is
    /// an immutable copy of that sequence's slot-state placed in the pinned
    /// stager `generation` (device pointer stable for the whole forward),
    /// instead of the live `gpu_chunks` buffer (which reallocates on the next
    /// chunk append). Used by the metadata builder so per-token snapshots built
    /// up front survive later chunk-boundary rebuilds. See
    /// [`super::gpu_chunks::GpuChunks::snapshot_into_generation`].
    pub fn sync_decode_gpu_chunks_snapshot(
        &self,
        batch_entries: &[(usize, usize)],
        arena_info: &[crate::kv_cache::arena_table::ResolvedArenaInfo],
        generation: &candle::quantized::pinned_staging::Generation,
        snapshot_mask: &[bool],
    ) -> candle::Result<(Vec<(u64, u32, u32)>, DecodeGpuChunkSyncStats)> {
        let n_kv_head = self.inner.n_kv_head;
        let head_dim = self.inner.head_dim;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let mut results = Vec::with_capacity(batch_entries.len());
        let mut stats = DecodeGpuChunkSyncStats::default();
        // A speculative-verify wave carries one entry per BLOCK POSITION, so a
        // sequence appears `block_len` times with an identical (seq, offset)
        // key — and an identical snapshot. Snapshot once per key and reuse the
        // copy for the duplicate rows (the state lock is held across the loop,
        // so nothing can mutate the slot buffer between duplicates).
        let mut snap_cache: HashMap<(usize, usize), (u64, u32, u32)> = HashMap::new();
        for (i, &(seq_idx, seq_offset)) in batch_entries.iter().enumerate() {
            let t_sync = std::time::Instant::now();
            // Per-entry: a sequence that mutates the arena during this forward
            // (a prefill absorbing across chunk boundaries, a glue gap-scatter)
            // needs an IMMUTABLE snapshot copy so its `slices_ptr` survives the
            // reallocation; a plain decode row never reallocs (its write chunk is
            // pre-ensured), so it keeps the zero-copy LIVE pointer + on-device
            // write-len commit — the cheap Qwen/Llama decode path.
            let want_snapshot = snapshot_mask.get(i).copied().unwrap_or(true);
            if want_snapshot {
                if let Some(&cached) = snap_cache.get(&(seq_idx, seq_offset)) {
                    stats.reuses += 1;
                    results.push(cached);
                    continue;
                }
            }
            let (result, sync_kind) = if let Some(Some(seq)) = state.sequences.get_mut(seq_idx) {
                seq.validate_decode_state(seq_idx, seq_offset)?;
                let ((live_ptr, n_slices, write_slice), kind) =
                    seq.sync_decode_gpu_chunks(n_kv_head, head_dim, seq_offset, arena_info)?;
                let ptr = if want_snapshot {
                    let ptr = seq.snapshot_gpu_chunks_into(generation, seq_offset)?;
                    snap_cache.insert((seq_idx, seq_offset), (ptr, n_slices, write_slice));
                    ptr
                } else {
                    live_ptr
                };
                ((ptr, n_slices, write_slice), kind)
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

impl BackingInner {
    /// Validate a selection batch's gids against the CURRENT storage arenas
    /// before any per-head table is uploaded and the selection kernel launched.
    ///
    /// The kernel addresses `row[gid / stride].base + (gid % stride) *
    /// row.chunk_byte_stride` with no bounds knowledge, so two host-state
    /// corruptions become silent OOB reads (or garbage-KV reads) on the GPU:
    ///   - a gid whose arena is ABSENT from storage (freed while referenced;
    ///     its table row is zeroed → near-null deref), and
    ///   - a gid whose `chunk_idx` exceeds the arena's per-FORMAT capacity
    ///     (`arena_chunks_for_format`): the arena index was re-tenanted with a
    ///     different-capacity format between gid mint and table build, so
    ///     `old_chunk_idx × new_stride` walks past the slab (the sanitizer-
    ///     confirmed read at exactly slab end).
    ///
    /// Failing here converts the corrupt launch into a named, recoverable
    /// error carrying both sides of the mismatch.
    #[cfg(feature = "cuda")]
    pub fn validate_selection_gids(&self, chunks: &[ChunkBands], label: &str) -> Result<()> {
        let stride = GID_STRIDE;
        let n_kv_head = self.n_kv_head;
        self.storage.read(|s| -> Result<()> {
            for (ci, hg) in chunks.iter().map(|c| &c.gids).enumerate() {
                for h in 0..n_kv_head {
                    for p in 0..N_PALETTE {
                        for (side, raw) in [
                            ("K", hg.k_gid_pal(h, p).raw()),
                            ("V", hg.v_gid_pal(h, p).raw()),
                        ] {
                            if raw < 0 {
                                continue; // sentinel / absent palette slot
                            }
                            let arena_idx = raw as usize / stride;
                            let chunk_idx = raw as usize % stride;
                            let Some(arena) = s.arena(arena_idx) else {
                                candle::bail!(
                                    "selection gid validation ({label}): chunk {ci} head {h} \
                                     palette {p} {side}-gid {raw} → arena {arena_idx} ABSENT \
                                     from storage (chunk_idx {chunk_idx}) — arena freed under \
                                     a live gid"
                                );
                            };
                            // Capacity is a property of the arena's SIZE CLASS,
                            // and a region re-stamped to a different class has a
                            // different stride and a different slot count. This
                            // reads like format bookkeeping and is actually a
                            // memory-safety net: it caught a sanitizer-confirmed
                            // OOB read at slab end (audit A14).
                            let cap = arena.chunks();
                            if chunk_idx >= cap {
                                candle::bail!(
                                    "selection gid validation ({label}): chunk {ci} head {h} \
                                     palette {p} {side}-gid {raw} → arena {arena_idx} is class \
                                     {} B (capacity {cap}) but chunk_idx is {chunk_idx} — \
                                     pool/storage class mismatch: the gid was minted under a \
                                     different-capacity class (arena index re-tenanted under \
                                     a live gid)",
                                    arena.slot_stride()
                                );
                            }
                        }
                    }
                }
            }
            Ok(())
        })?
    }

    /// Host-side build of the selection table: one [`PerHeadEntry`] row per
    /// `(chunk, head)` of `chunks`, in that order, each `PerHeadEntry::COLS`
    /// wide. The kernel indexes it as `chunk_idx * n_kv_head + head_idx`.
    ///
    /// **Every palette sub-entry is populated from that band's own gid** — its
    /// arena's base pointer and slot stride, and the *chunk's* own format tag.
    /// This is what lets bands of one head live in different arenas and
    /// different formats: the address and the layout both travel per band,
    /// so nothing about the row depends on the bands sharing an arena.
    ///
    /// Sizing follows the job list rather than storage, which is also what
    /// makes the cross-layer persist selection cheap: layers concatenate with
    /// no arena-index rebasing, and the table grows with chunks actually being
    /// selected instead of with the arena count (`docs/archived/arena_unification.md`
    /// E5). An empty job list yields an empty table.
    ///
    /// The whole build happens under ONE storage read — never two. A split read
    /// (bound under one lock, pointers under another) lets the scheduler
    /// free or relocate an arena in between, so the rows no longer describe the
    /// arenas the caller's frozen gids address, and the select kernel
    /// dereferences a stale base pointer
    /// (`run_select_kv_format_palette4_paged` → `CUDA_ERROR_ILLEGAL_ADDRESS`).
    /// **The single-lock capture is now the whole of that guarantee.** This
    /// used to credit a migrate-in-flight guard with blocking arena
    /// free/relocate/trim across the build→launch→readback window;
    /// `migrate_flight` is an advisory counter with no mutual exclusion, so it
    /// blocks nothing. What actually holds is that every pointer here comes from
    /// a pinned gid — the gid keeps its chunk's arena alive for the window — and
    /// the reservation makes an arena base permanently valid for the process,
    /// so there is no relocation to race with in the first place.
    // Host-side table builder: the selection kernels consume it under `cuda`,
    // and `selection_table_tests` pins its row layout with or without one.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub fn per_head_table_host(&self, chunks: &[ChunkBands]) -> Result<Vec<i64>> {
        use crate::kv_cache::arena_table::{PaletteSubEntry, N_PALETTE};

        let n_kv_head = self.n_kv_head;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        self.storage.read(|s| {
            let arenas = s.arenas();

            // Base pointer + slot stride per referenced arena, resolved once
            // up front. A chunk's eight bands usually hit one or two arenas,
            // but a whole job list hits each repeatedly — and resolving here
            // rather than inside the row loop is what lets a stride error
            // propagate instead of being swallowed into a zero.
            //
            // An arena that is absent from storage resolves to `(0, 0)`, the
            // same all-zero row the dense-over-storage build produced for a
            // hole. `validate_selection_gids` rejects that case before the
            // launch; this keeps the failure a named host error rather than a
            // near-null device deref if it ever slips through.
            let mut resolved: HashMap<usize, (u64, i64)> = HashMap::new();
            for chunk in chunks {
                for gid in chunk.gids.as_slice() {
                    let ai = gid.arena_idx();
                    if let std::collections::hash_map::Entry::Vacant(e) = resolved.entry(ai) {
                        let v = match arenas.get(&ai) {
                            // k_ptr == v_ptr: one combined K+V buffer per arena.
                            Some(a) => (a.base_ptr().unwrap_or(0), a.slot_stride() as i64),
                            None => (0, 0),
                        };
                        e.insert(v);
                    }
                }
            }
            let addr = |arena_idx: usize| -> (u64, i64) {
                resolved.get(&arena_idx).copied().unwrap_or((0, 0))
            };

            let mut data: Vec<i64> = vec![0i64; chunks.len() * n_kv_head * PerHeadEntry::COLS];
            for (ci, chunk) in chunks.iter().enumerate() {
                for h in 0..n_kv_head {
                    let palette: [PaletteSubEntry; N_PALETTE] = std::array::from_fn(|p| {
                        let (k_base, k_stride) = addr(chunk.gids.k_gid_pal(h, p).arena_idx());
                        let (v_base, v_stride) = addr(chunk.gids.v_gid_pal(h, p).arena_idx());
                        let (k_tag, v_tag) = chunk.band_tags(h, p);
                        PaletteSubEntry {
                            k_ptr: k_base,
                            v_ptr: v_base,
                            // The band's chunk index is applied kernel-side from
                            // its own gid, so the row carries only the base and
                            // the step. Offsets stay zero: a chunk slot *is* one
                            // (head, palette, side) band.
                            k_byte_offset: 0,
                            v_byte_offset: 0,
                            k_chunk_byte_stride: k_stride,
                            v_chunk_byte_stride: v_stride,
                            k_format_tag: ArenaFormatTag::from_u8(k_tag),
                            v_format_tag: ArenaFormatTag::from_u8(v_tag),
                            k_outer_scale: 1.0,
                            v_outer_scale: 1.0,
                        }
                    });
                    let off = (ci * n_kv_head + h) * PerHeadEntry::COLS;
                    data[off..off + PerHeadEntry::COLS]
                        .copy_from_slice(&PerHeadEntry { palette }.to_tensor_row());
                }
            }

            Ok(data)
        })?
    }

    /// Build a lightweight per-arena info snapshot for constructing persistent
    /// GPU slot buffers without building a full GPU tensor.
    ///
    /// Returns one [`ResolvedArenaInfo`] per arena index (dense, 0..num_arenas).
    /// The `num_arenas` count is determined by scanning all active GIDs.
    pub fn resolve_arena_info(&self) -> Result<Vec<ResolvedArenaInfo>> {
        self.resolve_arena_info_filtered(None)
    }

    /// Like [`Self::resolve_arena_info`] but only builds entries for the arena
    /// indices in `needed` (others stay at the zero default). The provenance
    /// pointer-resolve touches only a couple of arenas per scope, so skipping
    /// `to_arena_entry` for every other arena is the bulk of that path's cost.
    /// `None` resolves all arenas (the original behaviour).
    pub fn resolve_arena_info_for(
        &self,
        needed: &std::collections::HashSet<usize>,
    ) -> Result<Vec<ResolvedArenaInfo>> {
        self.resolve_arena_info_filtered(Some(needed))
    }

    fn resolve_arena_info_filtered(
        &self,
        needed: Option<&std::collections::HashSet<usize>>,
    ) -> Result<Vec<ResolvedArenaInfo>> {
        use crate::kv_cache::arena_table::ResolvedArenaInfo;

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
                    chunk_capacity: 0,
                };
                num_arenas
            ];

            for (&arena_idx, arena) in arenas.iter() {
                if let Some(needed) = needed {
                    if !needed.contains(&arena_idx) {
                        continue;
                    }
                }
                // Address stride and capacity only. A band's payload length is
                // the chunk's business, not the arena's — see
                // `ResolvedArenaInfo`'s "what is deliberately absent".
                info[arena_idx] = ResolvedArenaInfo {
                    base_ptr: arena.base_ptr().unwrap_or(0),
                    chunk_byte_stride: arena.slot_stride() as i64,
                    chunk_capacity: arena.chunks() as u32,
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

    /// GPU arena occupancy split float vs quant across the pool this backing
    /// shares (arenas are pooled globally across layers with the same head
    /// config, so one backing's view is the whole model's). Feeds the per-wave
    /// `kv-pool` diagnostic that validates the compress-to-free rung shrinks the
    /// float side. See [`GpuArenaClassStats`].
    pub fn gpu_arena_class_stats(&self) -> GpuArenaClassStats {
        self.inner.pool.gpu_class_stats()
    }

    /// True when at least one of `seq`'s chunks is still wholly in a GPU float /
    /// R16 source arena — i.e. `quantize_sealed_in_place` would do real kernel
    /// work on this sequence rather than pass it through unchanged.
    ///
    /// Mirrors the per-chunk eligibility test in `compress.rs`: a chunk is
    /// compressible when every one of its `(h, p)` source bands is GPU-resident
    /// and recorded as `Float` or `Quantized(R16)`. Used by the scheduler's
    /// compress-to-free relief rung to skip turns whose hot is already
    /// quantized (a prior relief pass, or the persistence thread, beat it to
    /// them), so an undrained `snapshot_pending_warm` backlog doesn't re-walk
    /// finished turns.
    ///
    /// The two halves of the test come from different owners: **location** is
    /// arena identity and is read from storage, **format** travels with the
    /// chunk and is read from its own band tags.
    #[cfg(feature = "cuda")]
    pub fn sealed_has_compressible_chunk(&self, seq: &SealedSequence) -> bool {
        self.inner
            .storage
            .read(|storage| {
                seq.chunks.iter().any(|chunk| {
                    chunk.bands().all(|(gid, tag)| {
                        super::chunk_ops::needs_reconcile_source_tag(tag)
                            && matches!(
                                storage.arena_key(gid.arena_idx()),
                                Some(k) if k.location == ArenaLocation::Gpu
                            )
                    })
                })
            })
            .unwrap_or(false)
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

        // A block counts as "quantized" if ANY of its bands is recorded in a
        // non-R16 quantized format. R16 is a raw storage format (K+Q capture),
        // not compression. The question is asked of the chunk's own tags — a
        // size-class arena has no format to report.
        let mut quantized_tokens = 0usize;
        let total_tokens = slot.block_count() * CHUNK_SIZE;
        for cw in slot.chunks_slice() {
            let any_quant = cw
                .bands()
                .any(|(_, tag)| tag.is_quantized() && tag != ArenaFormatTag::R16);
            if any_quant {
                quantized_tokens += CHUNK_SIZE;
            }
        }
        Ok((quantized_tokens, total_tokens))
    }

    /// Compute byte-level compression statistics for a sequence.
    /// Live band slots per size class for one sequence, plus how many of them
    /// are in formats **narrower than the smallest class**.
    ///
    /// This is audit A12's instrumentation. Splitting the 320 B rung into
    /// {64, 160, 320} would save 256 B/slot on `Q0`/`Q0_V`/`Q0_X` and
    /// 160 B/slot on `Q0_M2`/`Q1_S`, at the cost of two more classes whose
    /// steady-state partial tails run about half a region each — ≈16 MiB.
    /// Break-even is therefore ≈16 MiB ÷ ~200 B ≈ 65–84 K live slots in
    /// sub-320 formats, roughly **2 % of a ~4.8 M-slot pool** on this card.
    ///
    /// **Decision rule: split the low end only if sub-320 formats exceed ~2 %
    /// of live slots.** That fails at the C4/C5 production default, which never
    /// selects them, and is worth re-checking at C9/C10. This is what makes the
    /// rule answerable rather than a guess.
    ///
    /// Returns `(per-class live slots, sub-320 live slots)`.
    pub fn class_histogram(&self, batch_idx: usize) -> ([usize; SizeClass::COUNT], usize) {
        let mut per_class = [0usize; SizeClass::COUNT];
        let mut narrow = 0usize;
        let elems = self.inner.elems_per_chunk();
        let smallest = SizeClass::at(0).bytes();

        let Ok(state) = self.state.read() else {
            return (per_class, narrow);
        };
        let Some(slot) = state.sequences.get(batch_idx).and_then(|s| s.as_ref()) else {
            return (per_class, narrow);
        };
        for cw in slot.chunks_slice() {
            for (_, tag) in cw.bands() {
                let Some(payload) = payload_bytes_for_tag(tag, elems) else {
                    continue;
                };
                if let Some(class) = class_for_payload(payload) {
                    per_class[class.index()] += 1;
                }
                if payload < smallest {
                    narrow += 1;
                }
            }
        }
        (per_class, narrow)
    }

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

        // Walk every (head, palette, K/V) band individually. Each covers
        // CHUNK_SIZE × (head_dim / N_PALETTE) elements. Float and R16 bands are
        // excluded from both numerator and denominator so the ratio is
        // CR = 16 / effective_bpe over compressed bands only.
        let mut actual = 0f64;
        let mut n_quant = 0usize;
        for cw in slot.chunks_slice() {
            for (_, tag) in cw.bands() {
                let Some(KvFormat::Quantized(qf)) = tag.to_kv_format() else {
                    continue;
                };
                if qf == QuantFormat::R16 {
                    continue;
                }
                actual += qf.bits_per_elem() as f64 * cw.usage as f64;
                n_quant += cw.usage as usize;
            }
        }
        Ok((actual, n_quant))
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

        // Bands are interleaved [K_head0_pal0, V_head0_pal0, ...]; stride by 2
        // from 0 (K) or 1 (V) so only one side is counted. Each band reports the
        // format its own tag records.
        let start = usize::from(is_value);
        for cw in slot.chunks_slice() {
            for (_, tag) in cw.bands().skip(start).step_by(2) {
                let Some(fmt) = tag.to_kv_format() else {
                    continue;
                };
                let dtype = match fmt {
                    KvFormat::Float(dt) => dt.to_ggml_dtype(),
                    KvFormat::Quantized(qf) => qf.to_ggml_dtype(),
                };
                // One count per band slot, matching the per-block diagnostics
                // that show N_PALETTE × n_kv_head slots per format.
                *(ret.entry(dtype).or_default()) += 1;
            }
        }
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
        let elems = self.inner.elems_per_chunk();

        // Snapshot every band as `(arena_idx, chunk_idx, tag)` while holding the
        // state lock briefly. The tag travels with the gid because it is the
        // only thing that says how to decode the slot's bytes — a size-class
        // arena is untyped (`docs/archived/arena_unification.md` principle 8).
        //
        // IMPORTANT: all N_PALETTE sub-bands are dumped, not just palette 0.
        // Dumping one palette would emit a quarter of each head and quietly
        // invalidate the offline selection analysis this feeds.
        type Band = (usize, usize, ArenaFormatTag);
        let block_bands: Vec<Vec<(Band, Band)>> = {
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
                    // `bands()` yields K,V interleaved in slot order, which is
                    // head-major then palette-major — the layout this dump
                    // reconstructs.
                    let flat: Vec<Band> = cw
                        .bands()
                        .map(|(g, tag)| (g.arena_idx(), g.chunk_idx(), tag))
                        .collect();
                    flat.chunks_exact(2).map(|kv| (kv[0], kv[1])).collect()
                })
                .collect()
        };
        if block_bands.is_empty() {
            return Ok(vec![]);
        }

        let mut result = Vec::with_capacity(block_bands.len());
        for (block_idx, bands) in block_bands.iter().enumerate() {
            let maybe_kv: Option<(Vec<f32>, Vec<f32>)> = self.inner.storage.read(|s| {
                let arenas = s.arenas();
                let mut k_all = Vec::with_capacity(elems * bands.len());
                let mut v_all = Vec::with_capacity(elems * bands.len());
                for &((k_ai, k_ci, k_tag), (v_ai, v_ci, v_tag)) in bands.iter() {
                    k_all.extend(read_float_band(arenas.get(&k_ai)?, k_ci, k_tag, elems)?);
                    v_all.extend(read_float_band(arenas.get(&v_ai)?, v_ci, v_tag, elems)?);
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
    ///
    /// ```text
    /// sub_head_dim = head_dim / N_PALETTE blocks of 128 bytes each
    /// block[d] = { F16 d[32]  // K values for tokens 0..32 at dim d
    ///            , u16 q[32]  // F16 Q values for tokens 0..32 at dim d
    ///            }
    /// ```
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
        type Band = (usize, usize, ArenaFormatTag);
        let block_gids: Vec<(usize, Vec<(Band, Band)>)> = {
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
                    // Each band carries its own tag: the arena it points into is
                    // an untyped byte slab and cannot say whether the slot holds
                    // R16 or a float.
                    let flat: Vec<Band> = cw
                        .bands()
                        .map(|(g, tag)| (g.arena_idx(), g.chunk_idx(), tag))
                        .collect();
                    (
                        lo + i,
                        flat.chunks_exact(2).map(|kv| (kv[0], kv[1])).collect(),
                    )
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
                for &((k_ai, k_ci, k_tag), (v_ai, v_ci, v_tag)) in head_gids.iter() {
                    // K + Q come from an R16 band; the band's own tag says so.
                    let Some(k_arena) = arenas.get(&k_ai) else {
                        ok = false;
                        return Ok::<(), candle::Error>(());
                    };
                    if k_tag != ArenaFormatTag::R16 {
                        ok = false;
                        return Ok(());
                    }
                    // Ranged read: only this slot's bytes leave VRAM.
                    let Ok(view) = k_arena.slot_bytes(k_ci, r16_bytes_per_chunk) else {
                        ok = false;
                        return Ok(());
                    };
                    let chunk_bytes = view.to_vec1::<u8>()?;
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

                    // V is float in R16 mode.
                    let Some(v_arena) = arenas.get(&v_ai) else {
                        ok = false;
                        return Ok(());
                    };
                    match read_float_band(v_arena, v_ci, v_tag, elems_per_subband) {
                        Some(head_data) => v_all.extend(head_data),
                        None => {
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
        type ProbeBand = (usize, usize, ArenaFormatTag);
        let block_gids: Vec<(usize, Vec<(ProbeBand, ProbeBand)>)> = {
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
                    // The tag rides along with the gid: whether a band is R16 or
                    // F16 is recorded on the chunk, and the arena it points into
                    // is an untyped byte slab that cannot answer.
                    let flat: Vec<(usize, usize, ArenaFormatTag)> = cw
                        .bands()
                        .map(|(g, tag)| (g.arena_idx(), g.chunk_idx(), tag))
                        .collect();
                    (
                        lo + i,
                        flat.chunks_exact(2)
                            .map(|kv| (kv[0], kv[1]))
                            .collect::<Vec<_>>(),
                    )
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
            let is_r16_f16 = head_gids.iter().all(|&((_, _, k_tag), (_, _, v_tag))| {
                k_tag == ArenaFormatTag::R16 && v_tag == ArenaFormatTag::F16
            });
            if !is_r16_f16 {
                continue;
            }

            // Guard: both K (R16) and V (Float F16) arenas must be GPU-backed.
            // CPU-backed arenas have base_ptr=0; passing 0+ci*stride to the kernel
            // produces a small non-zero address that CUDA maps to unallocated memory
            // → ILLEGAL_ADDRESS fault on the GPU stream.
            let all_ptrs_nonzero = head_gids.iter().all(|&((k_ai, ..), (v_ai, ..))| {
                arena_info.get(k_ai).is_some_and(|a| a.base_ptr != 0)
                    && arena_info.get(v_ai).is_some_and(|a| a.base_ptr != 0)
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
            for &((k_ai, k_ci, _), (v_ai, v_ci, _)) in head_gids.iter() {
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

    /// Resolve the R16 K-chunk base pointers (Q is co-located in the K chunk at
    /// +64 per dim-group) for a sequence's blocks — the addresses the provenance
    /// sign-pack kernel reads Q from. Returns `(q_ptrs, block_indices)` where
    /// `q_ptrs` is `[block][head][palette]` device i64 addresses (length
    /// `block_indices.len() × n_kv_head × N_PALETTE`) and `block_indices` are the
    /// absolute chunk indices kept — only fully-R16, GPU-backed blocks are
    /// included, exactly as [`Self::gather_r16_kv_probe`] filters them, so all
    /// layers of a scope resolve to the same block set. Empty on non-CUDA / no
    /// R16 blocks.
    pub fn provenance_q_ptrs(
        &self,
        batch_idx: usize,
        block_range: Option<(usize, usize)>,
    ) -> Result<(Vec<i64>, Vec<usize>)> {
        use crate::kv_cache::arena_table::ArenaFormatTag;

        let Device::Cuda(_) = &self.inner.device else {
            return Ok((Vec::new(), Vec::new()));
        };

        // (absolute_block_idx, [(k_ai, k_ci)]) — Q side only (co-located with K).
        let block_gids: Vec<(usize, Vec<(usize, usize, ArenaFormatTag)>)> = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let slot = match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                Some(s) => s,
                None => return Ok((Vec::new(), Vec::new())),
            };
            let chunks = slot.chunks_slice();
            let (lo, hi) = match block_range {
                Some((l, h)) => (l.min(chunks.len()), h.min(chunks.len())),
                None => (0, chunks.len()),
            };
            if hi <= lo {
                return Ok((Vec::new(), Vec::new()));
            }
            chunks[lo..hi]
                .iter()
                .enumerate()
                .map(|(i, cw)| {
                    // K side only (Q is co-located), each band with its own tag.
                    let gids: Vec<(usize, usize, ArenaFormatTag)> = cw
                        .bands()
                        .step_by(2)
                        .map(|(g, tag)| (g.arena_idx(), g.chunk_idx(), tag))
                        .collect();
                    (lo + i, gids)
                })
                .collect()
        };
        if block_gids.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        // Resolve only the arenas this scope's gids actually reference (a couple),
        // not the whole arena table — the bulk of this path's per-scope cost.
        let needed: std::collections::HashSet<usize> = block_gids
            .iter()
            .flat_map(|(_, gids)| gids.iter().map(|&(k_ai, ..)| k_ai))
            .collect();
        let arena_info = self.inner.resolve_arena_info_for(&needed)?;
        let mut q_ptrs: Vec<i64> = Vec::new();
        let mut block_indices: Vec<usize> = Vec::new();
        for (block_idx, head_gids) in &block_gids {
            // Keep a block only when every (h,p) K side is R16 and GPU-backed —
            // same filter as `gather_r16_kv_probe`, so layers stay aligned. The
            // R16-ness is the chunk answering; the residency is the arena.
            let ok = head_gids.iter().all(|&(k_ai, _, k_tag)| {
                k_tag == ArenaFormatTag::R16
                    && arena_info.get(k_ai).is_some_and(|a| a.base_ptr != 0)
            });
            if !ok {
                continue;
            }
            block_indices.push(*block_idx);
            for &(k_ai, k_ci, _) in head_gids.iter() {
                let k_arena = &arena_info[k_ai];
                q_ptrs.push(k_arena.base_ptr as i64 + k_ci as i64 * k_arena.chunk_byte_stride);
            }
        }
        Ok((q_ptrs, block_indices))
    }

    /// Launch the provenance sign-pack kernel over R16 Q-chunk pointers spanning
    /// ALL layers of a scope, returning the packed sign bits (`n_warps ×
    /// CHUNK_SIZE` u32, warp-major: `out[warp*CHUNK_SIZE + token]` has bit `d`
    /// set iff Q dim `d` of that sub-band is `>= 0`). `all_q_ptrs` is the
    /// concatenation of every layer's [`Self::provenance_q_ptrs`] output; all
    /// layers share this backing's device, so any backing can launch for the
    /// whole set. One HtoD (pointers) + one launch + one DtoH (packed bits) —
    /// replaces the per-layer f16 K/Q/V round-trips. CUDA only.
    #[cfg(feature = "cuda")]
    pub fn run_prov_sign_pack(&self, all_q_ptrs: &[i64], sub_head_dim: usize) -> Result<Vec<u32>> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::kernels;

        let Device::Cuda(cuda_dev) = &self.inner.device else {
            return Ok(Vec::new());
        };
        if all_q_ptrs.is_empty() || sub_head_dim == 0 || sub_head_dim > 32 {
            return Ok(Vec::new());
        }
        let n_warps = all_q_ptrs.len();
        let out_len = n_warps * CHUNK_SIZE;

        // Reuse the grow-only device scratch: realloc only when this batch is
        // bigger than any prior one, so the steady-state seal pays no device
        // alloc / pointer memcpy_stod — only one HtoD into the existing buffer.
        let mut guard = self
            .inner
            .prov_sign_scratch
            .lock()
            .map_err(|_| candle::Error::Msg("prov_sign_scratch mutex poisoned".into()))?;
        let grow = guard
            .as_ref()
            .is_none_or(|s| s.ptrs.len() < n_warps || s.out.len() < out_len);
        if grow {
            let ptrs_cap = guard
                .as_ref()
                .map_or(n_warps, |s| s.ptrs.len().max(n_warps));
            let out_cap = guard.as_ref().map_or(out_len, |s| s.out.len().max(out_len));
            let ptrs = unsafe { cuda_dev.alloc::<i64>(ptrs_cap)? };
            let out = unsafe { cuda_dev.alloc::<u32>(out_cap)? };
            *guard = Some(ProvSignScratch { ptrs, out });
        }
        let scratch = guard.as_mut().unwrap();
        let stream = cuda_dev.cuda_stream();

        // One HtoD of the pointers into the reused buffer.
        cuda_dev.memcpy_htod(all_q_ptrs, &mut scratch.ptrs.slice_mut(..n_warps))?;

        let ptrs_view = scratch.ptrs.slice(..n_warps);
        let out_view = scratch.out.slice(..out_len);
        {
            let (pp, _pg) = ptrs_view.device_ptr(&stream);
            let (op, _og) = out_view.device_ptr(&stream);
            candle::set_kernel_breadcrumb("run_prov_sign_pack", file!(), line!());
            unsafe {
                kernels::simple::prov_sign_pack::run_prov_sign_pack(
                    pp as *const i64,
                    op as *mut std::ffi::c_void,
                    n_warps as i32,
                    sub_head_dim as i32,
                    stream.cu_stream() as *mut _,
                );
            }
        }

        // DtoH only the packed bits into a fresh host Vec — host allocation is
        // cheap; the reused device buffers are what mattered.
        let mut out = vec![0u32; out_len];
        stream
            .memcpy_dtoh(&out_view, &mut out)
            .map_err(candle::Error::wrap)?;
        stream.synchronize().map_err(candle::Error::wrap)?;
        Ok(out)
    }

    /// Non-CUDA stub — no GPU kernel, so the provenance capture uses its CPU path.
    #[cfg(not(feature = "cuda"))]
    pub fn run_prov_sign_pack(
        &self,
        _all_q_ptrs: &[i64],
        _sub_head_dim: usize,
    ) -> Result<Vec<u32>> {
        Ok(Vec::new())
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

    /// Borrow the backing's device. Used by call sites that need to
    /// hand a `&Device` to one of the kernel-driving helpers without
    /// threading it down from the caller.
    pub fn device(&self) -> &Device {
        &self.inner.device
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

/// Release every fully-empty arena across all registered backings, returning
/// their regions to the pool. Returns arenas freed.
///
/// The reactive twin of the per-backing sweep, callable from the region layer
/// itself: the transient-tier placement (`region_pool::place_transient`)
/// measures its footprint against *claimed* regions, and a region whose arena
/// went chunk-empty since the last periodic sweep still reads as claimed —
/// measured on the Llama-2 MHA gate, 238 empty arenas stood between a 4-region
/// tier and its ground. An arena release is a free-list push, so this is safe
/// wherever a claim is.
///
/// Called from `region_pool`, which is CUDA-only — there is no transient tier
/// to place against without a device.
#[cfg(feature = "cuda")]
pub(super) fn global_release_empty_arenas() -> usize {
    let mut freed = 0;
    if let Ok(registry) = BACKING_REGISTRY.lock() {
        for weak in registry.iter() {
            if let Some(inner) = weak.upgrade() {
                // The pool's atomic fast path makes a no-op backing cost one
                // load, so this is callable from every claim without weight.
                if inner.pool.has_reclaimable() {
                    freed += inner.release_empty_arenas().unwrap_or(0);
                }
            }
        }
    }
    freed
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
