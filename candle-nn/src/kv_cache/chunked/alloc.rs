//! Internal allocation methods for ChunkedKvBacking.
//!
//! This module contains methods for:
//! - Ensuring max block capacity
//! - Creating arenas
//! - Allocating chunks from free lists or new arenas
//! - Ensuring chunks are allocated for token writes

use std::cmp;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

#[allow(unused_imports)]
use candle::quantized::QTensor;
use candle::{DType, Device, Result, Tensor};

use super::backing::{request_global_compact, ChunkedKvBacking};
#[cfg(feature = "cuda")]
use super::backing::KV_DEVICE_OOM_MARKER;
use super::gid_pool::ChunkGid;
use super::head_gids::{HeadGids, GIDS_PER_HEAD};
use super::types::{arena_chunks_for_format, ChunkWindow, CHUNK_SIZE};
use super::{Arena, ArenaLocation};
use crate::kv_cache::arena_table::N_PALETTE;
use crate::kv_cache::chunked::backing::BackingInner;
use crate::kv_cache::chunked::ArenaStorageState;
use crate::kv_cache::{KvFormat, QuantFormat};

/// Bytes of device VRAM kept free so allocating a new KV arena never eats the
/// headroom the in-flight forward pass needs for activations. On WDDM the
/// driver would otherwise silently spill the over-budget allocation to host
/// memory and collapse GPU throughput; reserving headroom (and failing fast
/// when it can't be met) keeps everything resident.
///
/// The forward-pass activation working set is *transient* and *chunked*: ragged
/// prefill processes one `CHUNK_SIZE`-bounded slice at a time and decode is a
/// single token, so the peak scales with `batch × chunk × model_width` — a few
/// hundred MiB for the models we run — and is *independent of total VRAM*. The
/// previous `max(8% of total, 1 GiB)` heuristic instead scaled with card size,
/// so on a memory-tight card (16 GB mostly filled by an 11.7 GB model) it
/// reserved ~1.3 GiB — a third of the KV budget — and rejected KV allocations
/// that fit fine in VRAM with no spill. A fixed headroom matches what the
/// forward pass actually needs.
///
/// Default 256 MiB; override with `CANDLE_KV_VRAM_RESERVE_MB` (raise it for very
/// wide models such as 235B, where the per-chunk activation is larger).
#[cfg(feature = "cuda")]
fn vram_reserve_bytes(_total: usize) -> usize {
    if let Ok(v) = std::env::var("CANDLE_KV_VRAM_RESERVE_MB") {
        if let Ok(mb) = v.trim().parse::<usize>() {
            return mb.saturating_mul(1024 * 1024);
        }
    }
    // 384 MiB. Serves two roles: headroom for forward-pass activations in our
    // own accounting, AND the hard floor of *current* driver-free VRAM we refuse
    // to dip below (see `vram_has_room`). The floor is what keeps the machine
    // off the cliff — it guarantees we never drive free toward zero and push the
    // OS into a paging death-spiral, regardless of what our pool accounting
    // estimates. Sized to clear a 30B-A3B forward pass while still letting the
    // 16 GB dev card hold large session batches; override with
    // `CANDLE_KV_VRAM_RESERVE_MB` (e.g. raise it on a busy shared desktop).
    384 * 1024 * 1024
}

/// Whether `want` bytes can be allocated on `device` without pushing *our own*
/// live GPU footprint past the memory actually available to us. Returns `true`
/// (permit) on non-CUDA devices, or when the pool/total queries are unavailable
/// — the gate only blocks when we can prove our footprint wouldn't fit.
///
/// We gate on CUDA's own accounting of our stream-ordered memory pool
/// (`pool_used_bytes` = model weights + KV + activations) against `init_free` —
/// the VRAM free at device creation, i.e. `total` minus our CUDA context,
/// before the model loaded. We deliberately do NOT gate on the driver's current
/// `free`: on WDDM that is polluted by pageable memory from other processes
/// (desktop / IDE / browser) which the OS evicts the instant our resident set
/// needs the room, so gating on it false-OOMs and swings run-to-run. Our pool
/// usage counts only us; `init_free` is the budget that's ours to spend.
#[cfg(feature = "cuda")]
fn vram_has_room(device: &Device, want: usize) -> bool {
    let Device::Cuda(d) = device else {
        return true;
    };
    let gpu_id = match device.location() {
        candle::DeviceLocation::Cuda { gpu_id } => gpu_id,
        _ => return true,
    };
    let used = match d.pool_used_bytes() {
        Ok(u) => u,
        Err(_) => return true,
    };
    let (free, total) = match d.mem_get_info() {
        Ok(ft) => ft,
        Err(_) => return true,
    };
    let reserve = vram_reserve_bytes(total);
    // Stable budget: our own pool usage (model + KV + activations), against the
    // VRAM that was ours to spend — `init_free` = total minus the CUDA context,
    // captured at device creation with the pageable desktop excluded. This is
    // the primary gate and does NOT read the volatile driver free, so it never
    // false-OOMs or swings run-to-run. Falls back to total if unrecorded.
    let ceiling = candle::gpu_memory::device_init_free(gpu_id).unwrap_or(total);
    let budget_ok = used.saturating_add(want).saturating_add(reserve) <= ceiling;
    // Bytes this allocation must take *from the OS*, beyond what the pool already
    // holds reserved-but-free (`reserved - used`). cudarc's stream-ordered pool
    // retains freed blocks, so reusing them (e.g. the quant arenas a KV seal
    // frees, immediately re-allocated) costs zero new OS memory — the driver's
    // `free` doesn't move. Without this, the floor below would blindly reject
    // pure pool reuse whenever `free` was already low.
    let reserved = match d.pool_reserved_bytes() {
        Ok(r) => r,
        Err(_) => used,
    };
    let os_needed = want.saturating_sub(reserved.saturating_sub(used));
    // Hard safety floor: only allocations that GROW our OS footprint are gated
    // against the driver's *current* free — those are what could drive free
    // toward zero and push the OS into a paging death-spiral (on WDDM, evicting
    // an active desktop over PCIe locks the system). Pure pool reuse can't, so
    // it's always permitted.
    let free_ok = os_needed == 0 || free.saturating_sub(os_needed) >= reserve;
    let ok = budget_ok && free_ok;
    if std::env::var("KV_BUDGET_DEBUG").is_ok() {
        let mb = |b: usize| b / (1024 * 1024);
        eprintln!(
            "[kv-budget] gpu{gpu_id} pool_used={} pool_reserved={} free={} want={} os_needed={} reserve={} ceiling={} budget_ok={budget_ok} free_ok={free_ok} -> {}",
            mb(used), mb(reserved), mb(free), mb(want), mb(os_needed), mb(reserve), mb(ceiling),
            if ok { "ALLOW" } else { "REJECT" }
        );
    }
    ok
}

/// Gate a GPU arena allocation of `arena_bytes` on the VRAM budget. When the
/// budget is exceeded: if `retry_after_compact`, force a global compaction and
/// re-check; if there's still no room, return a typed
/// [`KV_DEVICE_OOM_MARKER`] error (instead of letting `cuMemAlloc` spill).
/// `Ok(())` means it's safe to allocate. No-op on non-CUDA / CPU arenas.
#[allow(unused_variables)]
fn ensure_vram_budget(
    device: &Device,
    location: ArenaLocation,
    arena_bytes: usize,
    retry_after_compact: bool,
    what: &str,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if matches!(location, ArenaLocation::Gpu) && !vram_has_room(device, arena_bytes) {
            if retry_after_compact {
                let _ = request_global_compact();
            }
            if !vram_has_room(device, arena_bytes) {
                return Err(candle::Error::Msg(format!(
                    "{KV_DEVICE_OOM_MARKER}: {what} arena needs {arena_bytes} B but free VRAM \
                     (minus reserve) is insufficient after compaction"
                )));
            }
        }
    }
    Ok(())
}

static ARENA_STATS_ENABLED: OnceLock<bool> = OnceLock::new();
static ARENA_CREATE_COUNT: AtomicU64 = AtomicU64::new(0);
static ARENA_CREATE_TOTAL_NS: AtomicU64 = AtomicU64::new(0);

fn arena_stats_enabled() -> bool {
    *ARENA_STATS_ENABLED.get_or_init(|| std::env::var("KV_ARENA_STATS").is_ok())
}

fn record_arena_create(kind: &str, location: ArenaLocation, index: usize, elapsed_ns: u64) {
    if !arena_stats_enabled() {
        return;
    }
    let total_count = ARENA_CREATE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let total_ns = ARENA_CREATE_TOTAL_NS.fetch_add(elapsed_ns, Ordering::Relaxed) + elapsed_ns;
    let took_ms = elapsed_ns as f64 / 1_000_000.0;
    let total_ms = total_ns as f64 / 1_000_000.0;
    let avg_ms = total_ms / (total_count as f64);
    eprintln!(
        "[arena-create] kind={kind} location={location:?} index={index} took_ms={took_ms:.3} total_count={total_count} total_ms={total_ms:.3} avg_ms={avg_ms:.3}"
    );
}

fn push_unique_key(keys: &mut Vec<super::arena::ArenaKey>, key: super::arena::ArenaKey) {
    if !keys.iter().any(|k| k == &key) {
        keys.push(key);
    }
}

impl ChunkedKvBacking {
    /// Pre-create one arena for baseline formats and quant candidates, then
    /// mark those arena indices as protected so compaction never tombstones them.
    pub(super) fn warm_protected_arenas(
        &self,
        compression: Option<&super::CompressionPolicy>,
    ) -> Result<()> {
        let location = self.inner.storage.default_location();
        let mut keys = Vec::new();

        // Baseline warm set requested for runtime stability: F16 and R16.
        push_unique_key(
            &mut keys,
            super::arena::ArenaKey::uniform(KvFormat::Float(DType::F16), location),
        );
        push_unique_key(
            &mut keys,
            super::arena::ArenaKey::uniform(KvFormat::Quantized(QuantFormat::R16), location),
        );

        // Include the backing's default target formats.
        push_unique_key(
            &mut keys,
            super::arena::ArenaKey::uniform(self.inner.storage.k_format(), location),
        );
        push_unique_key(
            &mut keys,
            super::arena::ArenaKey::uniform(self.inner.storage.v_format(), location),
        );

        // Include quantized candidates used by the shared adaptive profile.
        if let Some(compression) = compression {
            let (k_candidates, v_candidates) =
                super::compression_policy::production_adaptive_candidates(
                    compression.compression_level,
                );
            for fmt in k_candidates.iter().chain(v_candidates.iter()) {
                if let KvFormat::Quantized(qf) = fmt {
                    push_unique_key(
                        &mut keys,
                        super::arena::ArenaKey::uniform(KvFormat::Quantized(*qf), location),
                    );
                }
            }
        }

        for key in keys {
            let arena_idx = self.inner.pool.register_arena(key.clone());
            self.inner.pool.protect_arena(arena_idx);
            self.ensure_arena_exists(arena_idx, key)?;
        }
        Ok(())
    }

    /// Ensure the backing can hold at least `required_max_blocks` blocks per sequence.
    pub(super) fn ensure_max_blocks(&self, required_max_blocks: usize) -> Result<()> {
        if required_max_blocks <= 1 {
            return Ok(());
        }
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        if required_max_blocks <= state.max_blocks {
            return Ok(());
        }

        let mut new_max_blocks = state.max_blocks;
        while new_max_blocks < required_max_blocks {
            new_max_blocks = cmp::max(required_max_blocks, new_max_blocks.saturating_mul(2));
            if new_max_blocks == 0 {
                new_max_blocks = required_max_blocks;
                break;
            }
        }

        state.max_blocks = new_max_blocks;

        Ok(())
    }

    pub(super) fn create_arena(
        &self,
        shape: (usize, usize, usize),
        format: KvFormat,
        location: ArenaLocation,
        index: usize,
        retry_after_compact: bool,
    ) -> Result<Arena> {
        self.inner
            .create_arena(shape, format, location, index, retry_after_compact)
    }
}

impl BackingInner {
    /// Create a single arena with the given format, location, and index.
    ///
    /// Each arena chunk stores exactly one head's one side (K or V).
    /// Float arenas have shape `(arena_chunks, chunk_size, head_dim)`.
    /// Quantized arenas have `arena_chunks * chunk_size * head_dim` elements.
    pub(super) fn create_arena(
        &self,
        shape: (usize, usize, usize),
        format: KvFormat,
        location: ArenaLocation,
        index: usize,
        retry_after_compact: bool,
    ) -> Result<Arena> {
        let device = match location {
            ArenaLocation::Gpu => &self.device,
            ArenaLocation::Cpu => &Device::Cpu,
        };

        match format {
            KvFormat::Float(dtype) => {
                self.create_float_arena(shape, dtype, device, location, index, retry_after_compact)
            }
            KvFormat::Quantized(qformat) => self.create_quantized_arena(
                shape,
                qformat,
                device,
                location,
                index,
                retry_after_compact,
            ),
        }
    }

    /// Create a float arena — each chunk stores one head's one side.
    /// Shape: `(arena_chunks, chunk_size, head_dim)` — 3-D tensor.
    fn create_float_arena(
        &self,
        shape: (usize, usize, usize),
        dtype: DType,
        device: &Device,
        location: ArenaLocation,
        index: usize,
        retry_after_compact: bool,
    ) -> Result<Arena> {
        let t0 = Instant::now();
        let data_shape = shape;
        // Budget gate: refuse (and compact-then-fail) rather than let the
        // driver spill this arena to host memory. See `ensure_vram_budget`.
        let arena_bytes = data_shape
            .0
            .saturating_mul(data_shape.1)
            .saturating_mul(data_shape.2)
            .saturating_mul(dtype.size_in_bytes());
        if let Err(e) =
            ensure_vram_budget(device, location, arena_bytes, retry_after_compact, "float")
        {
            record_arena_create("float", location, index, t0.elapsed().as_nanos() as u64);
            return Err(e);
        }
        let out = match Tensor::zeros(data_shape, dtype, device) {
            Ok(data) => Ok(Arena::Float {
                data,
                dtype,
                location,
                index,
            }),
            Err(e) if retry_after_compact => {
                let freed = request_global_compact();
                if freed > 0 {
                    let data = Tensor::zeros(data_shape, dtype, device)?;
                    Ok(Arena::Float {
                        data,
                        dtype,
                        location,
                        index,
                    })
                } else {
                    Err(e)
                }
            }
            Err(e) => Err(e),
        };
        record_arena_create("float", location, index, t0.elapsed().as_nanos() as u64);
        out
    }

    /// Create a quantized arena — each chunk stores one head's one side.
    /// Total elements: `arena_chunks * chunk_size * head_dim`.
    fn create_quantized_arena(
        &self,
        shape: (usize, usize, usize),
        qformat: crate::kv_cache::QuantFormat,
        device: &Device,
        location: ArenaLocation,
        index: usize,
        retry_after_compact: bool,
    ) -> Result<Arena> {
        let t0 = Instant::now();
        let k_ggml = qformat.to_ggml_dtype();
        let total_elems = shape.0 * shape.1 * shape.2; // arena_chunks * chunk_size * head_dim

        // Budget gate (see `create_float_arena`). Quantized byte size derives
        // from the ggml block layout, not a plain dtype width.
        let block_size = k_ggml.block_size().max(1);
        let arena_bytes = (total_elems / block_size).saturating_mul(k_ggml.type_size());
        if let Err(e) =
            ensure_vram_budget(device, location, arena_bytes, retry_after_compact, "quantized")
        {
            record_arena_create("quantized", location, index, t0.elapsed().as_nanos() as u64);
            return Err(e);
        }

        let make_data = || QTensor::zeros(total_elems, k_ggml, device);

        let out = match make_data() {
            Ok(data) => Ok(Arena::Quantized {
                data,
                format: qformat,
                location,
                index,
            }),
            Err(e) if retry_after_compact => {
                let freed = request_global_compact();
                if freed > 0 {
                    let data = make_data()?;
                    Ok(Arena::Quantized {
                        data,
                        format: qformat,
                        location,
                        index,
                    })
                } else {
                    Err(e)
                }
            }
            Err(e) => Err(e),
        };
        record_arena_create("quantized", location, index, t0.elapsed().as_nanos() as u64);
        out
    }
}

impl ChunkedKvBacking {
    /// Allocate a chunk with pre-acquired arena lock.
    /// This version takes a pre-acquired arenas lock to avoid deadlock when called
    /// from contexts that already hold the arena lock.
    pub(super) fn alloc_chunk_with_arenas(
        &self,
        arena_state: &mut ArenaStorageState,
        key: super::arena::ArenaKey,
    ) -> Result<ChunkGid> {
        let arena_chunks = arena_chunks_for_format(key.format);
        let sub_head_dim = (self.inner.head_dim / N_PALETTE).max(1);

        if let Some(gid) = self.inner.pool.allocate_for(key.clone()) {
            let arena_idx = gid.arena_idx();
            let arena_was_fresh = !arena_state.has_arena(arena_idx);
            if arena_was_fresh {
                let arena = self.create_arena(
                    (arena_chunks, CHUNK_SIZE, sub_head_dim),
                    key.format,
                    key.location,
                    arena_idx,
                    true,
                )?;
                arena_state.push_arena(arena, arena_idx, arena_chunks);
            } else {
                // Free-list reuse on an existing arena: the chunk's bytes are
                // whatever the prior tenant left. Zero them so the new tenant
                // (and any persist quantize pass that reads past token_count)
                // sees clean storage. Fresh arenas are already zero from
                // Tensor::zeros / QTensor::zeros at creation, so the
                // arena_was_fresh branch above skips the work.
                self.zero_recycled_chunk(arena_state, arena_idx, gid.chunk_idx())?;
            }
            return Ok(gid);
        }

        let arena_idx = self.inner.pool.register_arena(key.clone());
        if !arena_state.has_arena(arena_idx) {
            let arena = self.create_arena(
                (arena_chunks, CHUNK_SIZE, sub_head_dim),
                key.format,
                key.location,
                arena_idx,
                true,
            )?;
            arena_state.push_arena(arena, arena_idx, arena_chunks);
        }

        let gid = self
            .inner
            .pool
            .allocate_for(key)
            .expect("just registered arena, must have capacity");
        Ok(gid)
    }

    /// Zero one recycled chunk's bytes. Asynchronous on CUDA — the work
    /// is enqueued on the device's primary stream and the call returns
    /// once queued. Same-stream FIFO ordering guarantees the next reader
    /// of this chunk sees the zeros without an explicit fence.
    fn zero_recycled_chunk(
        &self,
        arena_state: &mut ArenaStorageState,
        arena_idx: usize,
        chunk_idx: usize,
    ) -> Result<()> {
        let Some(arena) = arena_state.arenas_mut().get_mut(&arena_idx) else {
            return Ok(());
        };
        #[cfg(feature = "cuda")]
        {
            let stream_owned = match &self.inner.device {
                candle::Device::Cuda(cuda_dev) => Some(cuda_dev.cuda_stream()),
                _ => None,
            };
            arena.zero_chunk_at(chunk_idx, stream_owned.as_ref())
        }
        #[cfg(not(feature = "cuda"))]
        {
            arena.zero_chunk_at(chunk_idx)
        }
    }

    /// ArenaKey for active (unfilled) K chunks.
    ///
    /// On CUDA we keep the fast R16 active-K path for the decode/prefill kernels.
    /// On CPU we keep active K chunks in float so partial-token writes and tests
    /// do not require block-aligned quantization on every append.
    pub(super) fn active_k_arena_key(&self) -> super::arena::ArenaKey {
        let location = self.inner.storage.default_location();
        match self.inner.storage.k_format() {
            KvFormat::Float(dtype) => {
                super::arena::ArenaKey::uniform(KvFormat::Float(dtype), location)
            }
            KvFormat::Quantized(_) if matches!(location, ArenaLocation::Gpu) => {
                super::arena::ArenaKey::uniform(
                    KvFormat::Quantized(crate::kv_cache::QuantFormat::R16),
                    location,
                )
            }
            KvFormat::Quantized(_) => {
                super::arena::ArenaKey::uniform(KvFormat::Float(candle::DType::F16), location)
            }
        }
    }

    /// ArenaKey for active (unfilled) V chunks — always float.
    pub(super) fn active_v_arena_key(&self) -> super::arena::ArenaKey {
        let dtype = match self.inner.storage.k_format() {
            KvFormat::Float(dtype) => dtype,
            KvFormat::Quantized(_) => candle::DType::F16,
        };
        let location = self.inner.storage.default_location();
        super::arena::ArenaKey::uniform(KvFormat::Float(dtype), location)
    }

    /// Allocate a full block's worth of flat chunks for the palette4 arenas.
    ///
    /// Returns a `ChunkWindow` with `GIDS_PER_HEAD * n_kv_head` GIDs (N_PALETTE per
    /// head × {K, V}).  Each chunk stores exactly `CHUNK_SIZE * (head_dim / N_PALETTE)`
    /// elements — one head, one palette sub-band, one side.
    ///
    /// HeadGids layout: `head * GIDS_PER_HEAD + palette * 2 + is_value`.
    pub(super) fn alloc_block_chunks(&self, usage: u32, offset: u16) -> Result<ChunkWindow> {
        let n = GIDS_PER_HEAD * self.inner.n_kv_head;
        let k_key = self.active_k_arena_key();
        let v_key = self.active_v_arena_key();
        let mut gids = Vec::with_capacity(n);
        // For each head, allocate N_PALETTE K GIDs then N_PALETTE V GIDs in palette order.
        for _h in 0..self.inner.n_kv_head {
            for _p in 0..N_PALETTE {
                gids.push(self.alloc_chunk_for_key(k_key.clone())?);
                gids.push(self.alloc_chunk_for_key(v_key.clone())?);
            }
        }

        Ok(ChunkWindow {
            gids: HeadGids::from_vec(gids),
            usage,
            offset,
            k_pal: self.inner.identity_pal.clone(),
            v_pal: self.inner.identity_pal.clone(),
            k_scale: self.inner.identity_scale.clone(),
            v_scale: self.inner.identity_scale.clone(),
        })
    }

    pub(super) fn alloc_chunk_for_key(
        &self,
        key: super::arena::ArenaKey,
    ) -> Result<super::gid_pool::ChunkGid> {
        self.inner.alloc_chunk_for_key(key)
    }

    /// Bulk variant of [`Self::alloc_chunk_for_key`] — allocates `n`
    /// GIDs of the same `key` while paying the per-format pool mutex
    /// and the per-arena storage write lock only **once each**
    /// (instead of `n` times). Used by the cold-load
    /// `alloc_sealed_blocks_bulk` path where a single layer can need
    /// ~600 GIDs of the same format.
    pub(super) fn alloc_chunks_for_key_bulk(
        &self,
        key: super::arena::ArenaKey,
        n: usize,
    ) -> Result<Vec<super::gid_pool::ChunkGid>> {
        self.inner.alloc_chunks_for_key_bulk(key, n)
    }

    pub(super) fn ensure_arena_exists(
        &self,
        arena_idx: usize,
        key: super::arena::ArenaKey,
    ) -> Result<()> {
        self.inner.ensure_arena_exists(arena_idx, key)
    }
}

impl BackingInner {
    /// Allocate a chunk of any format through the pool exclusively.
    ///
    /// The pool's lock-free refcount table is the single source of truth
    /// for "this slot is allocated" — `pool.allocate_for` performs the
    /// CAS-claim that flips it. Storage only has to ensure the physical
    /// arena tensor exists; no separate per-slot bookkeeping is needed.
    ///
    /// 1. Try `pool.allocate_for(key)` — reuses a freed slot from any arena of this format.
    /// 2. If no capacity: `pool.register_arena(key)` → create physical arena → retry.
    pub(super) fn alloc_chunk_for_key(
        &self,
        key: super::arena::ArenaKey,
    ) -> Result<super::gid_pool::ChunkGid> {
        if let Some(gid) = self.pool.allocate_for(key.clone()) {
            self.ensure_arena_exists(gid.arena_idx(), key)?;
            return Ok(gid);
        }

        let arena_idx = self.pool.register_arena(key.clone());
        self.ensure_arena_exists(arena_idx, key.clone())?;

        Ok(self
            .pool
            .allocate_for(key)
            .expect("just registered arena, must have capacity"))
    }

    /// Bulk allocator — mirrors [`Self::alloc_chunk_for_key`]'s
    /// register-on-exhaustion loop but in batch.
    ///
    /// Per pass:
    /// - **One** `pool.allocate_n_for(key, remaining)` returns up to
    ///   `remaining` GIDs; CAS-claim makes the refcount table
    ///   immediately authoritative, no follow-up bookkeeping required.
    /// - **One** `ensure_arena_exists` per unique arena index we
    ///   touched (cheap — the inner check is a `storage.read`).
    ///
    /// If the pool returned fewer GIDs than requested, the format's
    /// pool was exhausted — we register a fresh arena (one
    /// `register_arena + ensure_arena_exists` round) and re-enter the
    /// loop to fill the remainder. Same termination guarantee as the
    /// singular path.
    pub(super) fn alloc_chunks_for_key_bulk(
        &self,
        key: super::arena::ArenaKey,
        n: usize,
    ) -> Result<Vec<super::gid_pool::ChunkGid>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut out: Vec<super::gid_pool::ChunkGid> = Vec::with_capacity(n);
        while out.len() < n {
            let remaining = n - out.len();
            let batch = self.pool.allocate_n_for(key.clone(), remaining);
            if batch.is_empty() {
                // Pool exhausted — register a fresh arena and retry.
                let arena_idx = self.pool.register_arena(key.clone());
                self.ensure_arena_exists(arena_idx, key.clone())?;
                continue;
            }
            // Ensure every unique arena index we just got is materialised
            // in storage. Most calls hit the cheap `storage.read`-only
            // path because the arena already exists.
            let mut seen: ahash::HashSet<usize> =
                ahash::HashSet::with_capacity_and_hasher(4, ahash::RandomState::new());
            for gid in &batch {
                let ai = gid.arena_idx();
                if seen.insert(ai) {
                    self.ensure_arena_exists(ai, key.clone())?;
                }
            }
            out.extend(batch);
        }
        Ok(out)
    }

    /// Ensure that an arena exists at the given index in storage.
    /// Creates the arena if it does not exist yet.
    pub(super) fn ensure_arena_exists(
        &self,
        arena_idx: usize,
        key: super::arena::ArenaKey,
    ) -> Result<()> {
        let arena_chunks = arena_chunks_for_format(key.format);
        let sub_head_dim = (self.head_dim / N_PALETTE).max(1);
        let shape = (arena_chunks, CHUNK_SIZE, sub_head_dim);

        let exists = self.storage.read(|s| s.has_arena(arena_idx))?;
        if exists {
            return Ok(());
        }

        let arena = self.create_arena_for_key(shape, &key, arena_idx, true)?;
        self.storage.try_write(|s| {
            if !s.has_arena(arena_idx) {
                s.push_arena(arena, arena_idx, arena_chunks);
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Create an arena from an ArenaKey, dispatching to the appropriate constructor.
    /// `head_formats` provides per-head (k, v) format lists for per-head arenas.
    fn create_arena_for_key(
        &self,
        shape: (usize, usize, usize),
        key: &super::arena::ArenaKey,
        index: usize,
        retry_after_compact: bool,
    ) -> Result<Arena> {
        self.create_arena(shape, key.format, key.location, index, retry_after_compact)
    }
}

impl ChunkedKvBacking {
    /// Ensure that chunks needed to write `add` tokens at `offsets` are allocated.
    ///
    /// `offsets` must have exactly `batch_capacity()` elements, one per sequence slot.
    pub fn ensure_for_offsets(&self, offsets: &[usize], adds: &[usize]) -> Result<()> {
        let batch = self.batch_capacity();
        if offsets.len() != batch {
            candle::bail!(
                "offset count mismatch: got {} offsets for chunked backing batch {}",
                offsets.len(),
                batch
            )
        }
        if adds.len() != offsets.len() {
            candle::bail!(
                "ensure_for_offsets: {} adds for {} offsets",
                adds.len(),
                offsets.len()
            )
        }
        if adds.iter().all(|&a| a == 0) {
            return Ok(());
        }

        let mut required_max_blocks = 1usize;
        for (i, &off) in offsets.iter().enumerate() {
            let end_pos = off.saturating_add(adds[i]).saturating_sub(1);
            let need_blocks = (end_pos / CHUNK_SIZE) + 1;
            required_max_blocks = cmp::max(required_max_blocks, need_blocks);
        }
        self.ensure_max_blocks(required_max_blocks)?;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        let chunk_size = CHUNK_SIZE;
        for (b, &off) in offsets.iter().enumerate() {
            // Skip unallocated slots and slots with nothing to add.
            if state.sequences[b].is_none() || adds[b] == 0 {
                continue;
            }

            let end_pos = off.saturating_add(adds[b]).saturating_sub(1);
            let need_blocks = (end_pos / chunk_size) + 1;
            for blk in 0..need_blocks {
                if state.sequences[b].as_ref().unwrap().chunk_at(blk).is_none() {
                    let slot = state.sequences[b].as_mut().unwrap();
                    // Seal the current write target to full capacity.
                    // We are allocating a new block because the write range
                    // extends past the previous last block; the kernel will
                    // fill every remaining position in that block.
                    if let Some(last) = slot.last_chunk_mut() {
                        let cur_offset = last.offset;
                        let capacity = chunk_size - cur_offset as usize;
                        last.usage = capacity as u32;
                    }
                    // Push new block with 2*n_kv_head flat chunks.
                    let cw = self.alloc_block_chunks(0, 0)?;
                    let slot = state.sequences[b].as_mut().unwrap();
                    slot.push_chunk(cw);
                }
            }
        }

        // Writable-tail pass: for each allocated slot, ensure the last block
        // can accept new writes. Under the read-only projection model the
        // tail is unshared by construction (projection pushes a fresh active
        // chunk; closed-quant chunks force a fresh push on the first write),
        // so we never have to COW here — we just allocate a new block when
        // the current tail is full or in a closed-off quant arena.
        for b in 0..batch {
            if state.sequences[b].is_none() {
                continue;
            }
            let needs_new_block: Option<bool> = state.sequences[b].as_ref().and_then(|s| {
                let cw = s.last_chunk()?;
                let is_full = (cw.offset as usize + cw.usage as usize) >= CHUNK_SIZE;
                if is_full {
                    Some(true)
                } else {
                    // Check if ANY arena referenced by this block is quantized
                    // (can't append to quantized chunks).  R16 is excluded: it
                    // uses Quantized(_) format but IS directly writable by the
                    // decode scatter kernel (write_regs_to_r16).
                    let unique_arenas = cw.gids.unique_arena_indices();
                    let is_quantized = self
                        .inner
                        .storage
                        .read(|s| {
                            unique_arenas.iter().any(|&ai| {
                                s.arena_key(ai)
                                    .map(|k| match k.format {
                                        crate::kv_cache::KvFormat::Quantized(
                                            crate::kv_cache::QuantFormat::R16,
                                        ) => false,
                                        crate::kv_cache::KvFormat::Quantized(_) => true,
                                        _ => false,
                                    })
                                    .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false);
                    if is_quantized {
                        Some(true)
                    } else {
                        None // partial + float (or R16) → already writable
                    }
                }
            });
            if let Some(true) = needs_new_block {
                let cw = self.alloc_block_chunks(0, 0)?;
                let slot = state.sequences[b].as_mut().unwrap();
                slot.push_chunk(cw);
            }
        }

        Ok(())
    }

    /// Force-push a fresh empty writer chunk onto a slot's chunk list.
    ///
    /// Unlike [`ensure_for_offset`] / [`ensure_for_batch_entries`], this
    /// always pushes a new chunk regardless of whether the slot's tail
    /// is technically writable.  Used by cumulative section ingest:
    /// after `inject_sealed_at_tail` Arc-clones the prefix sections'
    /// substrate chunks onto a fresh scratch slot, the slot's tail is
    /// the last prefix section's partial chunk (shared with substrate).
    /// Writing into it would mutate bytes other holders see as
    /// immutable section content.  Pushing a fresh empty chunk here
    /// makes the slot's *write target* a writer-owned chunk; the
    /// shared partial sits read-only just before it, and the prefill
    /// kernel starts writing at chunk-internal position 0 of the new
    /// chunk (= logical position `prefix_token_count`).
    pub fn push_empty_writer_chunk(&self, batch_idx: usize) -> Result<()> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        let current_block_count = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            state
                .sequences
                .get(batch_idx)
                .and_then(|s| s.as_ref())
                .map(|s| s.block_count())
                .unwrap_or(0)
        };
        self.ensure_max_blocks(current_block_count + 1)?;
        let cw = self.alloc_block_chunks(0, 0)?;
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        if let Some(Some(slot)) = state.sequences.get_mut(batch_idx) {
            slot.push_chunk(cw);
            // The freshly-appended empty chunk is now the writer. Advance the
            // writer boundary to it so any partial sealed chunk before it sits
            // below the boundary — its empty tail (gap) is then excluded from
            // the writer region and never written to or attended.
            let new_idx = slot.block_count().saturating_sub(1);
            slot.set_writer_start_idx(new_idx);
            slot.invalidate_gpu_chunks();
        } else {
            candle::bail!("push_empty_writer_chunk: slot {} not allocated", batch_idx)
        }
        Ok(())
    }

    /// Ensure chunks for a sparse batch of `(batch_idx, offset)` entries.
    ///
    /// This is the partial-batch analogue of [`ensure_for_offsets`]. It acquires
    /// the chunked state write-lock once and applies allocation/tail-writability
    /// checks only for the provided sequence slots.
    pub fn ensure_for_batch_entries(&self, entries: &[(usize, usize)], add: usize) -> Result<()> {
        if entries.is_empty() || add == 0 {
            return Ok(());
        }

        let batch = self.batch_capacity();
        for &(batch_idx, _off) in entries.iter() {
            if batch_idx >= batch {
                candle::bail!(
                    "batch_idx {} out of range for chunked backing (capacity {})",
                    batch_idx,
                    batch
                )
            }
        }

        let chunk_size = CHUNK_SIZE;

        // Under cum_token addressing the slot's `state.offset` is the
        // sum of slice.usage — NOT chunk_count × CHUNK_SIZE.  We can't
        // use positional math (`(offset + add) / CHUNK_SIZE`) to count
        // needed chunks because partial-tail slices (from injected
        // prefix sections) make positional and cum_token indices
        // diverge.  Instead, compute how many additional chunks each
        // slot needs based on the actual remaining capacity in
        // existing chunks starting at the first empty (or last
        // partial) slice — the same `write_slice` rule used in
        // `slot_state.rs`.
        let mut alloc_plan: Vec<(usize, usize)> = Vec::with_capacity(entries.len());
        let mut required_max_blocks = 1usize;
        {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for &(batch_idx, _off) in entries.iter() {
                let (current_chunks, available) =
                    match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                        Some(slot) => {
                            let chunks = slot.chunks_slice();
                            if chunks.is_empty() {
                                (0usize, 0usize)
                            } else {
                                // Writer-owned region starts at
                                // `writer_start_idx` (set by the host).
                                // Available capacity = remaining slots in
                                // each chunk from that index onward.
                                let start = slot.writer_start_idx().min(chunks.len() - 1);
                                let avail: usize = chunks[start..]
                                    .iter()
                                    .map(|c| chunk_size - (c.offset as usize + c.usage as usize))
                                    .sum();
                                (chunks.len(), avail)
                            }
                        }
                        None => (0usize, 0usize),
                    };
                let needed_extra = add.saturating_sub(available);
                let additional_chunks = (needed_extra + chunk_size - 1) / chunk_size;
                let new_total_chunks = current_chunks + additional_chunks;
                required_max_blocks = cmp::max(required_max_blocks, new_total_chunks.max(1));
                alloc_plan.push((batch_idx, additional_chunks));
            }
        }
        self.ensure_max_blocks(required_max_blocks)?;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        for (batch_idx, additional_chunks) in alloc_plan {
            // Auto-allocate slot if needed (mirrors ensure_for_offset behavior).
            if state.sequences[batch_idx].is_none() {
                state.sequences[batch_idx] = Some(self.make_sequence_state()?);
            }
            for _ in 0..additional_chunks {
                let cw = self.alloc_block_chunks(0, 0)?;
                let slot = state.sequences[batch_idx].as_mut().unwrap();
                slot.push_chunk(cw);
                slot.invalidate_gpu_chunks();
            }
        }

        // Writable-tail pass for touched entries only.
        for &(batch_idx, _off) in entries.iter() {
            let needs_new_block: Option<bool> = state.sequences[batch_idx].as_ref().and_then(|s| {
                let cw = s.last_chunk()?;
                debug_assert!(
                    cw.gids.iter().all(|g| g.strong_count() <= cw.gids.len()),
                    "tail block must not be shared — fork should have copied it"
                );
                let is_full = (cw.offset as usize + cw.usage as usize) >= CHUNK_SIZE;
                if is_full {
                    Some(true)
                } else {
                    let unique_arenas = cw.gids.unique_arena_indices();
                    let is_quantized = self
                        .inner
                        .storage
                        .read(|s| {
                            unique_arenas.iter().any(|&ai| {
                                s.arena_key(ai)
                                    .map(|k| match k.format {
                                        crate::kv_cache::KvFormat::Quantized(
                                            crate::kv_cache::QuantFormat::R16,
                                        ) => false,
                                        crate::kv_cache::KvFormat::Quantized(_) => true,
                                        _ => false,
                                    })
                                    .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false);
                    if is_quantized {
                        Some(true)
                    } else {
                        None
                    }
                }
            });

            if let Some(true) = needs_new_block {
                let cw = self.alloc_block_chunks(0, 0)?;
                let slot = state.sequences[batch_idx].as_mut().unwrap();
                slot.push_chunk(cw);
                slot.invalidate_gpu_chunks();
            }
        }

        Ok(())
    }

    pub fn ensure_for_offset(&self, batch_idx: usize, offset: usize, add: usize) -> Result<()> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        if add == 0 {
            return Ok(());
        }

        let end_pos = offset.saturating_add(add).saturating_sub(1);
        let need_blocks = (end_pos / CHUNK_SIZE) + 1;
        self.ensure_max_blocks(need_blocks)?;

        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;

        // Auto-allocate slot if needed
        if state.sequences[batch_idx].is_none() {
            state.sequences[batch_idx] = Some(self.make_sequence_state()?);
        }

        for blk in 0..need_blocks {
            if state.sequences[batch_idx]
                .as_ref()
                .unwrap()
                .chunk_at(blk)
                .is_none()
            {
                // Under cum_token addressing we never bump the
                // previous tail's usage when allocating a new chunk.
                // See `ensure_for_batch_entries` for rationale.
                // Push new block with 2*n_kv_head flat chunks.
                let cw = self.alloc_block_chunks(0, 0)?;
                let slot = state.sequences[batch_idx].as_mut().unwrap();
                slot.push_chunk(cw);
            }
        }

        // Writable-tail pass: ensure the last block is a float block that can be
        // written to.  After a full block we need a new empty block.  Fork paths
        // copy partial tails at fork time, so shared tails should never reach this point.
        let needs_new_block: Option<bool> = state.sequences[batch_idx].as_ref().and_then(|s| {
            let cw = s.last_chunk()?;
            debug_assert!(
                cw.gids.iter().all(|g| g.strong_count() <= cw.gids.len()),
                "tail block must not be shared — fork should have copied it"
            );
            let is_full = (cw.offset as usize + cw.usage as usize) >= CHUNK_SIZE;
            if is_full {
                Some(true)
            } else {
                // Check if ANY arena referenced by this block is quantized
                // (can't append to quantized chunks).  R16 is excluded: it
                // uses Quantized(_) format but IS directly writable by the
                // decode scatter kernel (write_regs_to_r16).
                let unique_arenas = cw.gids.unique_arena_indices();
                let is_quantized = self
                    .inner
                    .storage
                    .read(|s| {
                        unique_arenas.iter().any(|&ai| {
                            s.arena_key(ai)
                                .map(|k| match k.format {
                                    crate::kv_cache::KvFormat::Quantized(
                                        crate::kv_cache::QuantFormat::R16,
                                    ) => false,
                                    crate::kv_cache::KvFormat::Quantized(_) => true,
                                    _ => false,
                                })
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);
                if is_quantized {
                    Some(true)
                } else {
                    None // partial + float (or R16) → already writable
                }
            }
        });

        if let Some(true) = needs_new_block {
            let cw = self.alloc_block_chunks(0, 0)?;
            let slot = state.sequences[batch_idx].as_mut().unwrap();
            slot.push_chunk(cw);
        }

        Ok(())
    }
}
