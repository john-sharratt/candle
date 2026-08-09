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

#[cfg(feature = "cuda")]
use super::backing::KV_DEVICE_OOM_MARKER;
use super::backing::{request_global_compact, ChunkedKvBacking};
use super::gid_pool::ChunkGid;
use super::head_gids::HeadGids;
use super::types::{arena_chunks_for_format, ChunkWindow, CHUNK_SIZE};
use super::{Arena, ArenaLocation};
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
/// Scales with the card (`total / 12`, floored at 384 MiB); override with
/// `CANDLE_KV_VRAM_RESERVE_MB`.
#[cfg(feature = "cuda")]
fn vram_reserve_bytes(total: usize) -> usize {
    if let Ok(v) = std::env::var("CANDLE_KV_VRAM_RESERVE_MB") {
        if let Ok(mb) = v.trim().parse::<usize>() {
            return mb.saturating_mul(1024 * 1024);
        }
    }
    // Serves two roles: headroom for the forward pass's transient activations in
    // our own accounting, AND the hard floor of *current* driver-free VRAM we
    // refuse to dip below (see `vram_has_room`) — the floor that keeps the machine
    // off the WDDM paging cliff.
    //
    // The activation peak is NOT "a few hundred MiB independent of total VRAM":
    // a wide ingest prefill (a fat cohort of sequences attending a long context)
    // materialises several GiB of transient tensors, and even a small decode alloc
    // needs a CONTIGUOUS block the fragmented pool free-list can't provide. A fixed
    // 384 MiB reserve let the pool grow until only ~384 MiB driver-free remained —
    // exactly where those allocations spill to host memory and collapse throughput
    // to 13–46 s/forward (the 62→65 GiB grind). The peak correlates with the
    // workload width, which in turn correlates with the card (you run wide cohorts
    // on big cards), so scale the reserve with `total`: `total / 12` is 6 GiB on a
    // 72 GiB card, 2.7 GiB on 32 GiB, ~1.3 GiB on a 16 GiB card — enough contiguous
    // driver-free that the wide prefill activations (and any decode alloc) stay
    // resident. Floored at 384 MiB for tiny cards; raise via the env var if a
    // workload's activations still spill (it caps the hot-KV budget in exchange,
    // which the ingest's evict-to-cold absorbs).
    (total / 12).max(384 * 1024 * 1024)
}

/// Extra VRAM headroom kept free ON TOP of [`vram_reserve_bytes`] that ONLY a
/// compress-to-free operation may allocate from — the scheduler's ordinary KV
/// growth can never touch it (see [`vram_budget_available`]).
///
/// It exists to break a deadlock: under extreme KV pressure the hot→warm
/// migration (and in-session seal) couldn't allocate the small *transient*
/// quantized scratch it needs to compress and free the much larger float
/// source — so `quantize_sealed_in_place` failed, VRAM stayed pinned, and
/// nothing could ever drain (can't free VRAM because freeing needs VRAM).
/// Reserving a dedicated slice that normal allocation must always leave free
/// guarantees the compress path — wrapped in an [`EvictionScope`] — always has
/// physical room to run and reclaim far more than it borrows.
///
/// Default 128 MiB (one compress arena is ~16 MiB; this covers several in
/// flight across the persistence + scheduler threads); override with
/// `CANDLE_KV_EVICTION_RESERVE_MB`.
#[cfg(feature = "cuda")]
fn eviction_reserve_bytes() -> usize {
    if let Ok(v) = std::env::var("CANDLE_KV_EVICTION_RESERVE_MB") {
        if let Ok(mb) = v.trim().parse::<usize>() {
            return mb.saturating_mul(1024 * 1024);
        }
    }
    128 * 1024 * 1024
}

#[cfg(feature = "cuda")]
thread_local! {
    /// Whether the current thread is inside a compress-to-free operation and
    /// may therefore allocate from the [`eviction_reserve_bytes`] slice.
    static EVICTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// RAII scope: while alive, arena allocations *on this thread* may dip into the
/// dedicated [`eviction_reserve_bytes`] headroom. Wrap a compress-to-free
/// operation (`quantize_sealed_in_place`) in it so the transient quantized
/// scratch it needs can always be allocated, even when the normal KV budget is
/// exhausted. Nestable — restores the previous state on drop.
#[cfg(feature = "cuda")]
pub(crate) struct EvictionScope(bool);

#[cfg(feature = "cuda")]
impl EvictionScope {
    pub(crate) fn enter() -> Self {
        EvictionScope(EVICTING.with(|c| c.replace(true)))
    }
}

#[cfg(feature = "cuda")]
impl Drop for EvictionScope {
    fn drop(&mut self) {
        EVICTING.with(|c| c.set(self.0));
    }
}

#[cfg(feature = "cuda")]
fn evicting() -> bool {
    EVICTING.with(|c| c.get())
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
    // Normal allocations must leave both the base reserve AND the dedicated
    // eviction slice free; a compress-to-free op (inside an `EvictionScope`) may
    // dip into the eviction slice — that's the whole point, so it can always
    // allocate its transient scratch and reclaim VRAM under extreme pressure.
    let reserve = vram_reserve_bytes(total)
        + if evicting() {
            0
        } else {
            eviction_reserve_bytes()
        };
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

/// Accurate "how many bytes of KV VRAM budget are free right now", for the
/// scheduler's budget-aware eviction.
///
/// When a [`VRAM Governor`](candle::vram) is installed, its **live measurement**
/// is the source of truth: on WDDM that is DXGI's per-process real free
/// (`Budget − CurrentUsage`), on Linux `cuMemGetInfo` — neither of which lies the
/// way the legacy `init_free − pool_used` estimate did (it stayed positive while
/// the pool oversubscribed the card and WDDM paged, causing the 14–1300 s stalls).
/// Falls back to the legacy estimate only when no governor is installed. `None`
/// on non-CUDA / query failure (caller treats that as "unknown — don't evict").
#[cfg(feature = "cuda")]
pub fn vram_budget_available(device: &Device) -> Option<usize> {
    let Device::Cuda(d) = device else {
        return None;
    };
    let gpu_id = match device.location() {
        candle::DeviceLocation::Cuda { gpu_id } => gpu_id,
        _ => return None,
    };
    // Governor path: the honest, physically-resident free VRAM PLUS the pool's
    // reserved-but-free bytes. A new KV arena reuses a pool free-block with **no
    // new OS memory** — `reserved` doesn't move, DXGI's per-process usage doesn't
    // move — so those bytes are available too. Without counting them the governor
    // reported false pressure while 10+ GiB of reusable pool sat free (the pool
    // can't return it to the OS: freed chunks are scattered inside partially-used
    // segments, so `cuMemPoolTrimTo` reclaims nothing), evicting pointlessly and
    // collapsing the admission window.
    //
    // BUT this only holds while the pool has NOT claimed the whole card. Reuse is
    // genuine for chunk-sized KV arenas, which slot into the fragmented free-blocks;
    // it is a LIE for the large, CONTIGUOUS activation buffers each forward
    // allocates — those can't use scattered free-blocks, so they force the pool to
    // grow, and once `reserved >= capacity` (the balloon-measured resident C) that
    // growth spills to host memory (WDDM) and collapses throughput to 13–46 s/forward.
    // In that oversubscribed state the reuse is phantom headroom that keeps the
    // scheduler admitting into the wall (`budget` reads ~8 GiB while `used` flatlines
    // and free = 0). Drop it there so the budget honestly falls to `headroom` (≈0),
    // admission backs off, and relief fires instead of digging deeper. `capacity == 0`
    // means the balloon hasn't measured yet (startup, no pressure) — keep the reuse.
    if let Some(gov) = candle::vram::get(gpu_id) {
        let headroom = gov.measure().ok()?.headroom as usize;
        let (reserved, used) = match (d.pool_reserved_bytes(), d.pool_used_bytes()) {
            (Ok(r), Ok(u)) => (r, u),
            _ => (0, 0),
        };
        let capacity = gov.capacity() as usize;
        let reuse = if capacity == 0 || reserved < capacity {
            reserved.saturating_sub(used)
        } else {
            0
        };
        return Some(headroom.saturating_add(reuse));
    }
    // Legacy fallback: init_free − pool_used − reserve.
    let used = d.pool_used_bytes().ok()?;
    let (_free, total) = d.mem_get_info().ok()?;
    let reserve = vram_reserve_bytes(total) + eviction_reserve_bytes();
    let ceiling = candle::gpu_memory::device_init_free(gpu_id).unwrap_or(total);
    Some(ceiling.saturating_sub(used).saturating_sub(reserve))
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
        if let Err(e) = ensure_vram_budget(
            device,
            location,
            arena_bytes,
            retry_after_compact,
            "quantized",
        ) {
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
        let sub_head_dim = (self.inner.head_dim / self.inner.n_palette()).max(1);

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
        // Band count per head: LATENT_N_BANDS (single-latent) or N_PALETTE (GQA).
        let np = self.inner.n_palette();
        let n = np * 2 * self.inner.n_kv_head;
        let k_key = self.active_k_arena_key();
        let v_key = self.active_v_arena_key();
        // Rope-region key for the single latent: the 64 RoPE dims (bands
        // [LATENT_NOPE_BANDS, np)) are pinned BF16 regardless of the writer
        // format, matching the reference (`nope FP8 ‖ rope BF16`). When the
        // writer format is already BF16 (the wave window) this equals `k_key`,
        // so the store is uniform BF16 — the pre-existing behaviour, only the
        // arena width narrows to the single-latent band width.
        let rope_key = super::arena::ArenaKey::uniform(
            KvFormat::Float(DType::BF16),
            self.inner.storage.default_location(),
        );
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
        for _h in 0..self.inner.n_kv_head {
            if single_latent {
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
                let nope_run = self
                    .inner
                    .alloc_chunk_run_for_key(k_key.clone(), nope_bands)?;
                let rope_run = if rope_bands > 0 {
                    self.inner
                        .alloc_chunk_run_for_key(rope_key.clone(), rope_bands)?
                } else {
                    Vec::new()
                };
                for k_gid in nope_run.into_iter().chain(rope_run) {
                    let v_gid = k_gid.clone();
                    gids.push(k_gid);
                    gids.push(v_gid);
                }
            } else {
                let k_run = self.inner.alloc_chunk_run_for_key(k_key.clone(), np)?;
                let v_run = self.inner.alloc_chunk_run_for_key(v_key.clone(), np)?;
                for (k_gid, v_gid) in k_run.into_iter().zip(v_run) {
                    gids.push(k_gid);
                    gids.push(v_gid);
                }
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
            // Fresh float writer chunk: transient, no resident record. The host
            // serializer builds per-forward scratch heads for it.
            meta: None,
        })
    }

    pub(super) fn alloc_chunk_for_key(
        &self,
        key: super::arena::ArenaKey,
    ) -> Result<super::gid_pool::ChunkGid> {
        self.inner.alloc_chunk_for_key(key)
    }

    /// See [`BackingInner::alloc_chunk_run_for_key`].
    pub(super) fn alloc_chunk_run_for_key(
        &self,
        key: super::arena::ArenaKey,
        len: usize,
    ) -> Result<Vec<super::gid_pool::ChunkGid>> {
        self.inner.alloc_chunk_run_for_key(key, len)
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

    /// Allocate `len` CONSECUTIVE slots in one arena of `key`. Contiguity is a
    /// LOCALITY optimization for the paged select/QREL walk, not a correctness
    /// requirement — each band is addressed through its own gid
    /// (`resolve_band_source`), so scattered bands read correctly, just with
    /// worse spatial locality. Falls back to singleton allocation when no run
    /// is available. Mirrors [`Self::alloc_chunk_for_key`]'s
    /// register-on-exhaustion retry.
    pub(super) fn alloc_chunk_run_for_key(
        &self,
        key: super::arena::ArenaKey,
        len: usize,
    ) -> Result<Vec<super::gid_pool::ChunkGid>> {
        if let Some(gids) = self.pool.allocate_run_for(key.clone(), len) {
            self.ensure_arena_exists(gids[0].arena_idx(), key)?;
            return Ok(gids);
        }
        let arena_idx = self.pool.register_arena(key.clone());
        self.ensure_arena_exists(arena_idx, key.clone())?;
        self.pool
            .allocate_run_for(key, len)
            .ok_or_else(|| candle::Error::Msg("fresh arena cannot fit palette run".into()))
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
        let sub_head_dim = (self.head_dim / self.n_palette()).max(1);
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

    /// Reserve an in-place **glue gap**: a fresh chunk of `n_tokens` valid slots
    /// appended at the slot tail, returning its block index. The gap's K/V is
    /// left uninitialised — the glue forward fills it via explicit `(slice,
    /// in_blk)` write targets, scattering before it streams, so nothing reads it
    /// unfilled.
    ///
    /// **The gap is full by construction.** It is allocated `offset =
    /// CHUNK_SIZE - n_tokens`, `usage = n_tokens`, so its valid window is the
    /// tail `[offset, CHUNK_SIZE)` and `offset + usage == CHUNK_SIZE`. This is
    /// the load-bearing invariant: a *partial* writer-owned chunk is, by the
    /// cache's own rules, an extendable writable tail — `extend_for_write_region`
    /// walks into it, `set_len` advances its usage, the writable-tail pass
    /// CoW-extends it. A *full* chunk is immutable to all of them: `write_slice`
    /// and `decode_write_chunk_idx` skip it, `set_len`'s cap is 0, `ensure`'s
    /// available-space sum counts it as 0, and the writable-tail pass pushes a
    /// fresh writer chunk instead of extending into it. The gap can therefore
    /// never be mistaken for the live writer region — which is exactly what makes
    /// the next prefill incapable of overflowing into it.
    ///
    /// `usage` is still exactly `n_tokens`, so the cumulative-usage `rope_base`
    /// of every later chunk equals its logical position by construction — the
    /// single positional convention the decode and glue kernels both read via
    /// `slice_rope` (a column's position is `slice_rope(c) + (in_blk - offset)`,
    /// so the tail window maps to `[rope_base, rope_base + n_tokens)`). The GIDs
    /// are unique (rc=1), so the glue's explicit write is safe and the next
    /// reproject's truncate frees them by refcount. `writer_start` is advanced
    /// PAST the gap so a subsequent sealed inject lands after it.
    ///
    /// Returns `(gap_block_index, in_blk_base)`, where `in_blk_base == offset` is
    /// the first valid slot of the tail window — the caller scatters the glue's
    /// K/V into `[in_blk_base, in_blk_base + n_tokens)` so the write lands exactly
    /// where this chunk's `slice_offset` expects it (no second, independent
    /// computation of the window can drift from the reservation).
    pub fn reserve_glue_gap_chunk(&self, batch_idx: usize, n_tokens: u32) -> Result<(usize, u32)> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        if n_tokens == 0 || n_tokens as usize > CHUNK_SIZE {
            candle::bail!(
                "reserve_glue_gap_chunk: n_tokens {} must be in 1..={CHUNK_SIZE}",
                n_tokens
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
        // +2: the immutable gap chunk PLUS an empty writable chunk placed after it
        // (see below).
        self.ensure_max_blocks(current_block_count + 2)?;
        // Full-by-construction: valid window is the chunk tail `[offset, 32)` with
        // `offset + usage == CHUNK_SIZE`, so the gap is immutable to every
        // writer-region scan (see the doc above).
        let offset = (CHUNK_SIZE as u32 - n_tokens) as u16;
        let cw = self.alloc_block_chunks(n_tokens, offset)?;
        // A fresh empty writer chunk to sit AFTER the gap. Without it the gap is
        // `last_chunk()`, and a co-batched decode/prefill on this same slot in the
        // unified wave validates its `last_chunk()` as the writable tail — the gap
        // is full-by-construction (`offset+usage == CHUNK_SIZE`), so the write-slice
        // check fails ("writable tail is already full/stale"), the wave forward
        // aborts, and the paged kernel is left to read a stale slot → illegal
        // address. Leaving an empty writable chunk past the gap keeps the decode's
        // `last_chunk()` a real writer (matching the write path, which already
        // targets `writer_start_idx`, not the gap). The crash root at 42553ca3.
        let writer_cw = self.alloc_block_chunks(0, 0)?;
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        if let Some(Some(slot)) = state.sequences.get_mut(batch_idx) {
            slot.push_chunk(cw);
            let gap_idx = slot.block_count().saturating_sub(1);
            // Advance the writer boundary PAST the gap so it is never the active
            // writer; the glue forward fills it by explicit target, and the next
            // sealed inject appends after it. The empty writer chunk we push next
            // is exactly at `gap_idx + 1`, so the boundary lands on a real tail.
            slot.set_writer_start_idx(gap_idx + 1);
            slot.push_chunk(writer_cw);
            slot.invalidate_gpu_chunks();
            Ok((gap_idx, offset as u32))
        } else {
            candle::bail!("reserve_glue_gap_chunk: slot {} not allocated", batch_idx)
        }
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
                            // Writer-owned capacity ONLY: chunks before
                            // `writer_start_idx` are Arc-shared/sealed, and a
                            // partial sealed tail is a GAP — its free slots
                            // are dead, never a write target. When the
                            // boundary sits past the last chunk (a freshly
                            // injected prefix), available is ZERO; clamping
                            // into the sealed tail counts its gap as writer
                            // capacity and under-allocates the write region
                            // by up to one chunk.
                            let start = slot.writer_start_idx();
                            let avail: usize = if start >= chunks.len() {
                                0
                            } else {
                                chunks[start..]
                                    .iter()
                                    .map(|c| chunk_size - (c.offset as usize + c.usage as usize))
                                    .sum()
                            };
                            (chunks.len(), avail)
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
