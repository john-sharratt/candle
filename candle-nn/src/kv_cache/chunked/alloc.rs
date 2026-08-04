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
/// Every term the arena budget gate weighs, kept together so a refusal can
/// report the WHOLE arithmetic rather than "insufficient".
///
/// A refusal used to say only "needs N B but free VRAM (minus reserve) is
/// insufficient after compaction" — which is unfalsifiable from a log: it names
/// neither the budget that was exceeded, nor by how much, nor which of the two
/// independent tests failed. Diagnosing one cost a full session; this struct is
/// what that session wished existed.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct VramGateFacts {
    /// Bytes the allocation asked for.
    pub want: usize,
    /// Our stream-ordered pool: bytes handed out, and bytes held from the OS.
    pub pool_used: usize,
    pub pool_reserved: usize,
    /// Bytes this allocation must take FROM THE OS beyond the pool's
    /// reserved-but-free gap (0 ⇒ pure pool reuse, always permitted).
    pub os_needed: usize,
    /// Driver-reported free / total on the device right now.
    pub free: usize,
    pub total: usize,
    /// The budget ceiling: VRAM that was ours at device creation (`init_free`),
    /// i.e. total minus the CUDA context, before the model loaded.
    pub ceiling: usize,
    /// Reserve withheld: the activation/floor term plus the eviction slice
    /// (waived inside an `EvictionScope`).
    pub reserve_base: usize,
    pub reserve_evict: usize,
    /// `pool_used + want + reserve <= ceiling` — our own footprint fits.
    pub budget_ok: bool,
    /// `os_needed == 0 || free - os_needed >= reserve` — growth won't drive the
    /// driver's free toward the WDDM paging cliff.
    pub free_ok: bool,
    /// When `!budget_ok`: how far past the ceiling the request would put us.
    pub budget_shortfall: usize,
    /// When `!free_ok`: how much more driver-free the request needed.
    pub free_shortfall: usize,
}

#[cfg(feature = "cuda")]
impl VramGateFacts {
    pub fn ok(&self) -> bool {
        self.budget_ok && self.free_ok
    }
    fn reserve(&self) -> usize {
        self.reserve_base + self.reserve_evict
    }
    /// One-line summary in MiB — the form both the log line and the error
    /// message carry, so a WARN and the propagated error agree exactly.
    fn summary(&self) -> String {
        let mb = |b: usize| b / (1024 * 1024);
        format!(
            "want={}MiB pool_used={}MiB pool_reserved={}MiB gap={}MiB os_needed={}MiB \
             free={}MiB total={}MiB ceiling(init_free)={}MiB reserve={}MiB(base {}+evict {}) \
             budget_ok={} free_ok={} budget_shortfall={}MiB free_shortfall={}MiB",
            mb(self.want),
            mb(self.pool_used),
            mb(self.pool_reserved),
            mb(self.pool_reserved.saturating_sub(self.pool_used)),
            mb(self.os_needed),
            mb(self.free),
            mb(self.total),
            mb(self.ceiling),
            mb(self.reserve()),
            mb(self.reserve_base),
            mb(self.reserve_evict),
            self.budget_ok,
            self.free_ok,
            mb(self.budget_shortfall),
            mb(self.free_shortfall),
        )
    }
}

/// The two gate tests, as pure arithmetic — split out so the decision is
/// testable against raw numbers without a CUDA device.
///
/// The two tests deliberately use DIFFERENT notions of cost, and the difference
/// is load-bearing:
///
/// - `free_ok` gates only GROWTH (`os_needed`), because an allocation served
///   from the pool's existing free blocks cannot move the driver's `free`.
/// - `budget_ok` gates the FULL `want` against the startup ceiling, **including
///   apparent reuse**, because `os_needed` is an optimistic lower bound.
///
/// That asymmetry looks like an oversight and is not. `os_needed` is derived
/// from the AGGREGATE gap (`reserved - used`), but the pool's free space is
/// fragmented: a contiguous 16 MiB arena can fail to fit in any single free
/// block while ~2 GiB sits free in aggregate, and the pool then grows its OS
/// reservation anyway. Measured by relaxing `budget_ok` for `os_needed == 0`:
/// `pool_reserved` grew 1312 MiB while `pool_used` grew only 428 MiB, the pool
/// ran 2480 MiB past its 13488 MiB capacity, driver-free hit 0, WDDM began
/// spilling, and prefill forwards went from ~1.1 s to 69 s.
///
/// So `budget_ok` is the wall that keeps the pool inside the card. Do not relax
/// it on an `os_needed == 0` fast path. The real remedy for a pool stuck at the
/// wall is to make it SHRINK — see the compaction path, which currently returns
/// `arenas_freed=0`.
///
/// Returns `(budget_ok, free_ok)`.
fn gate_decide(
    want: usize,
    used: usize,
    os_needed: usize,
    free: usize,
    ceiling: usize,
    reserve: usize,
) -> (bool, bool) {
    // Our own footprint against the VRAM that was ours at device creation.
    // Charges the full `want` — see the asymmetry note above.
    let budget_ok = used.saturating_add(want).saturating_add(reserve) <= ceiling;
    // Hard safety floor: only allocations that GROW our OS footprint are gated
    // against the driver's *current* free — those are what could drive free
    // toward zero and push the OS into a paging death-spiral (on WDDM, evicting
    // an active desktop over PCIe locks the system).
    let free_ok = os_needed == 0 || free.saturating_sub(os_needed) >= reserve;
    (budget_ok, free_ok)
}

/// Weigh the budget gate and return every term (see [`VramGateFacts`]).
/// `None` when the device isn't CUDA or the pool/total queries fail — the gate
/// only blocks when it can PROVE the footprint wouldn't fit, so an unknown
/// state permits.
#[cfg(feature = "cuda")]
fn vram_gate(device: &Device, want: usize) -> Option<VramGateFacts> {
    let Device::Cuda(d) = device else {
        return None;
    };
    let gpu_id = match device.location() {
        candle::DeviceLocation::Cuda { gpu_id } => gpu_id,
        _ => return None,
    };
    let used = d.pool_used_bytes().ok()?;
    let (free, total) = d.mem_get_info().ok()?;
    // Normal allocations must leave both the base reserve AND the dedicated
    // eviction slice free; a compress-to-free op (inside an `EvictionScope`) may
    // dip into the eviction slice — that's the whole point, so it can always
    // allocate its transient scratch and reclaim VRAM under extreme pressure.
    let reserve_base = vram_reserve_bytes(total);
    let reserve_evict = if evicting() {
        0
    } else {
        eviction_reserve_bytes()
    };
    let reserve = reserve_base + reserve_evict;
    // Stable budget: our own pool usage (model + KV + activations), against the
    // VRAM that was ours to spend — `init_free` = total minus the CUDA context,
    // captured at device creation with the pageable desktop excluded. This is
    // the primary gate and does NOT read the volatile driver free, so it never
    // false-OOMs or swings run-to-run. Falls back to total if unrecorded.
    let ceiling = candle::gpu_memory::device_init_free(gpu_id).unwrap_or(total);
    // Bytes this allocation must take *from the OS*, beyond what the pool already
    // holds reserved-but-free (`reserved - used`). cudarc's stream-ordered pool
    // retains freed blocks, so reusing them (e.g. the quant arenas a KV seal
    // frees, immediately re-allocated) costs zero new OS memory — the driver's
    // `free` doesn't move.
    let reserved = match d.pool_reserved_bytes() {
        Ok(r) => r,
        Err(_) => used,
    };
    let os_needed = want.saturating_sub(reserved.saturating_sub(used));
    let (budget_ok, free_ok) = gate_decide(want, used, os_needed, free, ceiling, reserve);
    let facts = VramGateFacts {
        want,
        pool_used: used,
        pool_reserved: reserved,
        os_needed,
        free,
        total,
        ceiling,
        reserve_base,
        reserve_evict,
        budget_ok,
        free_ok,
        budget_shortfall: if budget_ok {
            0
        } else {
            used.saturating_add(want)
                .saturating_add(reserve)
                .saturating_sub(ceiling)
        },
        free_shortfall: if free_ok {
            0
        } else {
            os_needed.saturating_add(reserve).saturating_sub(free)
        },
    };
    if std::env::var("KV_BUDGET_DEBUG").is_ok() {
        eprintln!(
            "[kv-budget] gpu{gpu_id} {} -> {}",
            facts.summary(),
            if facts.ok() { "ALLOW" } else { "REJECT" }
        );
    }
    Some(facts)
}

/// Whether `want` bytes fit — see [`vram_gate`]. Permits when the gate can't
/// measure (non-CUDA / query failure).
#[cfg(feature = "cuda")]
fn vram_has_room(device: &Device, want: usize) -> bool {
    vram_gate(device, want).map(|f| f.ok()).unwrap_or(true)
}

/// Bytes a NEW arena may still take before [`vram_has_room`] starts refusing —
/// the allocator's own remaining budget, computed from the exact same terms as
/// its `budget_ok` test so the two can never drift.
///
/// Admission MUST clamp to this. It is a different quantity from the governor's
/// availability (driver headroom + reuse gap), and on a card whose pool has
/// grown to its startup budget it is far smaller — the measured wedge was
/// `pool_used=12477 + reserve=1493 > ceiling=13488`, i.e. **−482 MiB** of real
/// arena budget while the governor still reported 3506 MiB of driver headroom.
/// Admission believed the headroom, kept admitting 7–8 sequences wide, and every
/// resulting arena creation was refused until the ingest hit its failure cap.
///
/// The two numbers are both honest and both needed: the governor's tracks
/// physical residency (what WDDM will spill), this one tracks our own footprint
/// against the VRAM that was ours at startup (what the allocator enforces).
/// Admission takes the MINIMUM.
///
/// `None` on non-CUDA devices or when the pool/total queries fail — the caller
/// treats that as "unknown", never as zero.
#[cfg(feature = "cuda")]
pub fn kv_alloc_headroom(device: &Device) -> Option<usize> {
    let Device::Cuda(d) = device else {
        return None;
    };
    let gpu_id = match device.location() {
        candle::DeviceLocation::Cuda { gpu_id } => gpu_id,
        _ => return None,
    };
    let used = d.pool_used_bytes().ok()?;
    let (_, total) = d.mem_get_info().ok()?;
    // Same reserve the gate applies to a normal (non-eviction) allocation.
    let reserve = vram_reserve_bytes(total) + eviction_reserve_bytes();
    let ceiling = candle::gpu_memory::device_init_free(gpu_id).unwrap_or(total);
    Some(ceiling.saturating_sub(used.saturating_add(reserve)))
}

#[cfg(not(feature = "cuda"))]
pub fn kv_alloc_headroom(_device: &Device) -> Option<usize> {
    None
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
            let before = vram_gate(device, arena_bytes);
            let reclaimed = if retry_after_compact {
                request_global_compact()
            } else {
                0
            };
            if let Some(after) = vram_gate(device, arena_bytes) {
                if !after.ok() {
                    // Report the WHOLE gate — both tests, both shortfalls, and
                    // what reclaim achieved — on the WARN and in the error text.
                    // The propagated message is what the ingest layer surfaces,
                    // so it has to stand alone without the log.
                    let detail = after.summary();
                    return Err(candle::Error::Msg(format!(
                        "{KV_DEVICE_OOM_MARKER}: {what} arena of {arena_bytes} B refused after \
                         reclaim(arenas_freed={reclaimed}) — {detail}"
                    )));
                }
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
        let (k, _) = crate::kv_cache::active_kv_formats(
            self.inner.storage.k_format(),
            matches!(location, ArenaLocation::Gpu),
        );
        match k {
            KvFormat::Float(dtype) => {
                super::arena::ArenaKey::uniform(KvFormat::Float(dtype), location)
            }
            KvFormat::Quantized(qf) => {
                super::arena::ArenaKey::uniform(KvFormat::Quantized(qf), location)
            }
        }
    }

    /// ArenaKey for active (unfilled) V chunks — always float.
    pub(super) fn active_v_arena_key(&self) -> super::arena::ArenaKey {
        let location = self.inner.storage.default_location();
        let (_, v) = crate::kv_cache::active_kv_formats(
            self.inner.storage.k_format(),
            matches!(location, ArenaLocation::Gpu),
        );
        super::arena::ArenaKey::uniform(v, location)
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
        // Per head: one CONTIGUOUS run of N_PALETTE K slots and one of V slots
        // (see `alloc_chunk_run_for_key`). Correctness does NOT depend on the
        // run layout — every kernel addresses each band through its own gid
        // (`resolve_band_source`, and the per-palette KvHead record pointers) —
        // but contiguous bands give the select/QREL walk better spatial
        // locality, so we mint them as runs where a run is available.
        // HeadGids layout stays `head * GIDS_PER_HEAD + palette * 2 + is_value`.
        for _h in 0..self.inner.n_kv_head {
            let k_run = self
                .inner
                .alloc_chunk_run_for_key(k_key.clone(), N_PALETTE)?;
            let v_run = self
                .inner
                .alloc_chunk_run_for_key(v_key.clone(), N_PALETTE)?;
            for (k_gid, v_gid) in k_run.into_iter().zip(v_run) {
                gids.push(k_gid);
                gids.push(v_gid);
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
        // A run larger than one arena can NEVER be satisfied — arenas are
        // fixed-capacity slabs and a run must be contiguous within one. Fail
        // with the sizes so this permanent condition is never mistaken for the
        // transient race below (they used to share one message, and a night went
        // into telling them apart).
        let arena_chunks = arena_chunks_for_format(key.format);
        if len > arena_chunks {
            candle::bail!(
                "palette run of {len} chunks exceeds arena capacity {arena_chunks} \
                 for format {:?} — cannot be satisfied by any arena",
                key.format,
            );
        }
        if let Some(gids) = self.pool.allocate_run_for(key.clone(), len) {
            self.ensure_arena_exists(gids[0].arena_idx(), key)?;
            return Ok(gids);
        }
        // No existing arena has tail room: register a fresh one and claim from
        // it BY INDEX. The old shape — register, then re-walk the whole pool —
        // raced: between registration and the re-walk, concurrent claimers (the
        // scheduler's prefills and the persistence thread's elevations allocate
        // the same formats in parallel) could consume the fresh arena's tail,
        // and the single retry then failed spuriously as "fresh arena cannot
        // fit palette run", killing the whole forward. Targeting the registered
        // index removes the which-arena race; losing even that (racers landing
        // in OUR arena via their own global walks) just means another
        // registration, bounded.
        const ATTEMPTS: usize = 4;
        for _ in 0..ATTEMPTS {
            let arena_idx = self.pool.register_arena(key.clone());
            self.ensure_arena_exists(arena_idx, key.clone())?;
            if let Some(gids) = self.pool.allocate_run_for_in(key.clone(), arena_idx, len) {
                return Ok(gids);
            }
            // Raced into our fresh arena — the racer may equally have vacated
            // tail room elsewhere; check the whole pool before registering again.
            if let Some(gids) = self.pool.allocate_run_for(key.clone(), len) {
                self.ensure_arena_exists(gids[0].arena_idx(), key)?;
                return Ok(gids);
            }
        }
        candle::bail!(
            "palette run of {len} chunks unsatisfied after {ATTEMPTS} fresh arenas \
             (capacity {arena_chunks} each, format {:?}) — allocator contention or \
             VRAM exhaustion on arena creation",
            key.format,
        )
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

        let chunk_size = CHUNK_SIZE;

        // Count first, allocate WITHOUT the guard, then install. `alloc_block_chunks`
        // can reach `request_global_compact`, which needs a write guard on every
        // layer's block table; allocating under one made that a self-deadlock and,
        // once the compactor was made non-blocking, a permanent no-op — so arena
        // compaction could never run from the prefill path that needs it most.
        // See `ensure_for_batch_entries` for the full rationale.
        // Plan blocks AND predict tail needs in one read pass, allocate both with
        // no guard held, then mutate under ONE write guard — extending a sequence
        // and making its tail writable must be atomic (a reader seeing blocks
        // pushed but the tail unreplaced would write into a full or closed-quant
        // chunk). Predicting before the installs over-estimates safely: a freshly
        // pushed block is writable, so installs only shrink the need.
        let mut plan: Vec<(usize, usize)> = Vec::new();
        let mut tail_maybe: Vec<usize> = Vec::new();
        {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for (b, &off) in offsets.iter().enumerate() {
                if state.sequences[b].is_none() || adds[b] == 0 {
                    continue;
                }
                let end_pos = off.saturating_add(adds[b]).saturating_sub(1);
                let need_blocks = (end_pos / chunk_size) + 1;
                let slot = state.sequences[b].as_ref().unwrap();
                let missing = (0..need_blocks)
                    .filter(|&blk| slot.chunk_at(blk).is_none())
                    .count();
                if missing > 0 {
                    plan.push((b, missing));
                }
            }
            for b in 0..batch {
                if state.sequences[b].is_some() && self.tail_needs_new_block(&state, b) {
                    tail_maybe.push(b);
                }
            }
        }
        let mut prealloc: Vec<(usize, Vec<_>)> = Vec::with_capacity(plan.len());
        for (b, missing) in plan {
            let mut cws = Vec::with_capacity(missing);
            for _ in 0..missing {
                cws.push(self.alloc_block_chunks(0, 0)?);
            }
            prealloc.push((b, cws));
        }
        let mut tail_spares: Vec<(usize, _)> = Vec::with_capacity(tail_maybe.len());
        for b in tail_maybe {
            tail_spares.push((b, self.alloc_block_chunks(0, 0)?));
        }

        // Fast path — see `ensure_for_batch_entries`.
        if prealloc.is_empty() && tail_spares.is_empty() {
            return Ok(());
        }

        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for (b, cws) in prealloc {
                let Some(slot) = state.sequences[b].as_mut() else {
                    continue;
                };
                for cw in cws {
                    // Seal the current write target to full capacity. We are
                    // allocating a new block because the write range extends past
                    // the previous last block; the kernel will fill every remaining
                    // position in that block. The seal must stay immediately before
                    // its push.
                    if let Some(last) = slot.last_chunk_mut() {
                        let cur_offset = last.offset;
                        let capacity = chunk_size - cur_offset as usize;
                        last.usage = capacity as u32;
                    }
                    slot.push_chunk(cw);
                }
            }
            // Writable-tail pass, same guard.
            for (b, cw) in tail_spares {
                if state.sequences[b].is_some() && self.tail_needs_new_block(&state, b) {
                    let slot = state.sequences[b].as_mut().unwrap();
                    slot.push_chunk(cw);
                }
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
    /// Whether [`Self::ensure_for_batch_entries`] would allocate anything —
    /// a read-only predicate over this layer's block table.
    ///
    /// Exists so the all-layers form can decide ONCE instead of per layer.
    pub fn batch_entries_need_work(&self, entries: &[(usize, usize)], add: usize) -> Result<bool> {
        if add == 0 {
            return Ok(false);
        }
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        for &(batch_idx, _off) in entries.iter() {
            if batch_idx >= state.sequences.len() {
                return Ok(true); // out of range -> let the real call report it
            }
            let Some(slot) = state.sequences[batch_idx].as_ref() else {
                return Ok(true); // slot needs allocating
            };
            let available = slot
                .last_chunk()
                .map(|cw| CHUNK_SIZE - (cw.offset as usize + cw.usage as usize).min(CHUNK_SIZE))
                .unwrap_or(0);
            if available < add || self.tail_needs_new_block(&state, batch_idx) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Ensure EVERY layer's backing has capacity for the upcoming write.
    ///
    /// Hoisted out of the per-layer decode loop deliberately. The block
    /// structure is layer-invariant — every layer is extended by this same call
    /// with identical `entries`/`add`, which is why the decode path already
    /// builds its position map from layer 0 and applies it to all layers. So the
    /// PLAN is computed once, from the first backing; only when work is genuinely
    /// needed does this touch all of them.
    ///
    /// Doing the plan per layer cost 48 lock acquisitions and 48 tail-predicate
    /// passes **per decoded token** — the steady state is "nothing to allocate",
    /// so that was almost entirely wasted work on the hot path.
    pub fn ensure_for_batch_entries_all(
        backings: &[ChunkedKvBacking],
        entries: &[(usize, usize)],
        add: usize,
    ) -> Result<()> {
        let Some(first) = backings.first() else {
            return Ok(());
        };
        if !first.batch_entries_need_work(entries, add)? {
            debug_assert!(
                backings
                    .iter()
                    .all(|b| !b.batch_entries_need_work(entries, add).unwrap_or(true)),
                "block structure must be layer-invariant: layer 0 needs no work but another layer does"
            );
            return Ok(());
        }
        for b in backings {
            b.ensure_for_batch_entries(entries, add)?;
        }
        Ok(())
    }

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
        let mut tail_maybe: Vec<usize> = Vec::new();
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
                // Predict the tail need in the SAME read pass — this runs once per
                // layer per decode step, so a second guard acquisition here is
                // pure overhead on the hot path.
                if self.tail_needs_new_block(&state, batch_idx) {
                    tail_maybe.push(batch_idx);
                }
            }
        }
        self.ensure_max_blocks(required_max_blocks)?;

        // Allocate BEFORE taking the state guard.
        //
        // `alloc_block_chunks` can reach `request_global_compact`, and compaction
        // needs a write guard on EVERY layer's block table to remap relocated
        // GIDs. Allocating while holding one of those guards made that a
        // self-deadlock; once the compactor was made non-blocking it became a
        // permanent no-op instead, so arena compaction could never run from the
        // path that needs it most. The chunk counts are already known from
        // `alloc_plan`, so the allocation needs no state access at all.
        let mut prealloc: Vec<(usize, Vec<_>)> = Vec::with_capacity(alloc_plan.len());
        for (batch_idx, additional_chunks) in alloc_plan {
            let mut cws = Vec::with_capacity(additional_chunks);
            for _ in 0..additional_chunks {
                cws.push(self.alloc_block_chunks(0, 0)?);
            }
            prealloc.push((batch_idx, cws));
        }

        // Predict which tails will need a fresh block, so their chunks can be
        // allocated alongside the rest and ALL mutation can happen under ONE
        // guard. The prediction is a safe over-estimate: installing a block makes
        // that sequence's tail fresh and therefore writable, so the block installs
        // below can only ever *reduce* this set, never grow it. Spares that turn
        // out to be unnecessary simply drop, returning their GIDs to the pool.
        let mut tail_spares: Vec<(usize, _)> = Vec::with_capacity(tail_maybe.len());
        for batch_idx in tail_maybe {
            tail_spares.push((batch_idx, self.alloc_block_chunks(0, 0)?));
        }

        // FAST PATH. This runs 48x per decode step (once per layer), and on a
        // normal step the block is not full and the tail is still writable, so
        // there is nothing to install. Returning before the write guard keeps the
        // steady-state decode cost at one read guard, as it was before the
        // allocation was hoisted out of the mutation guard.
        if prealloc.is_empty() && tail_spares.is_empty() {
            return Ok(());
        }

        // Single guard for every mutation. Extending a sequence and making its
        // tail writable must be ATOMIC: a reader that observes blocks pushed but
        // the tail not yet replaced would write into a full or closed-quant chunk.
        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for (batch_idx, cws) in prealloc {
                // Auto-allocate slot if needed (mirrors ensure_for_offset behavior).
                if state.sequences[batch_idx].is_none() {
                    state.sequences[batch_idx] = Some(self.make_sequence_state()?);
                }
                let slot = state.sequences[batch_idx].as_mut().unwrap();
                for cw in cws {
                    slot.push_chunk(cw);
                    slot.invalidate_gpu_chunks();
                }
            }

            // Writable-tail pass, still under the same guard.
            for (batch_idx, cw) in tail_spares {
                if self.tail_needs_new_block(&state, batch_idx) {
                    let slot = state.sequences[batch_idx].as_mut().unwrap();
                    slot.push_chunk(cw);
                    slot.invalidate_gpu_chunks();
                }
            }
        }

        Ok(())
    }

    /// Whether `batch_idx`'s tail block can still be written into, or a fresh
    /// block must be pushed. Pure read over state + arena storage; extracted so
    /// the decision can be made under a guard while the allocation it implies
    /// happens outside one.
    fn tail_needs_new_block(
        &self,
        state: &super::types::BlockTableState,
        batch_idx: usize,
    ) -> bool {
        let needs: Option<bool> = state.sequences[batch_idx].as_ref().and_then(|s| {
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
        matches!(needs, Some(true))
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

        // Count first, allocate WITHOUT the guard, then install. `alloc_block_chunks`
        // can reach `request_global_compact`, which needs a write guard on every
        // layer's block table; allocating under one made that a self-deadlock.
        // See `ensure_for_batch_entries` for the full rationale.
        // Count blocks AND predict the tail need in one read pass, allocate both
        // without a guard, then mutate under ONE write guard. Extending a sequence
        // and making its tail writable must be atomic: a reader that saw blocks
        // pushed but the tail not yet replaced would write into a full or
        // closed-quant chunk. Predicting the tail before the installs is a safe
        // over-estimate — a freshly pushed block is itself writable, so installs
        // can only shrink the need. An unused spare drops, returning its GIDs.
        let (missing, maybe_tail) = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let have = state.sequences[batch_idx]
                .as_ref()
                .map(|s| {
                    (0..need_blocks)
                        .filter(|&b| s.chunk_at(b).is_some())
                        .count()
                })
                .unwrap_or(0);
            (
                need_blocks.saturating_sub(have),
                self.tail_needs_new_block(&state, batch_idx),
            )
        };
        let mut fresh = Vec::with_capacity(missing);
        for _ in 0..missing {
            fresh.push(self.alloc_block_chunks(0, 0)?);
        }
        let tail_spare = if maybe_tail {
            Some(self.alloc_block_chunks(0, 0)?)
        } else {
            None
        };

        // Fast path — see `ensure_for_batch_entries`.
        if fresh.is_empty() && tail_spare.is_none() {
            return Ok(());
        }

        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            // Auto-allocate slot if needed
            if state.sequences[batch_idx].is_none() {
                state.sequences[batch_idx] = Some(self.make_sequence_state()?);
            }
            {
                let slot = state.sequences[batch_idx].as_mut().unwrap();
                // Under cum_token addressing we never bump the previous tail's
                // usage when allocating a new chunk. See `ensure_for_batch_entries`.
                for cw in fresh {
                    if (0..need_blocks).any(|b| slot.chunk_at(b).is_none()) {
                        slot.push_chunk(cw);
                    }
                }
            }
            // Writable-tail pass, same guard.
            if let Some(cw) = tail_spare {
                if self.tail_needs_new_block(&state, batch_idx) {
                    let slot = state.sequences[batch_idx].as_mut().unwrap();
                    slot.push_chunk(cw);
                }
            }
        }

        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod gate_decide_tests {
    use super::gate_decide;

    const MIB: usize = 1024 * 1024;

    /// Apparent pure reuse is STILL charged against the ceiling.
    ///
    /// This looks over-strict — `os_needed == 0` says the pool's 1,097 MiB gap
    /// covers the request — but the gap is AGGREGATE, not contiguous. Relaxing
    /// exactly this case was measured: `pool_reserved` grew 1312 MiB against
    /// only 428 MiB of `pool_used` (allocations billed as free reuse were taking
    /// new OS memory), the pool ran 2480 MiB past its 13488 MiB capacity,
    /// driver-free hit 0, WDDM began spilling, and prefill forwards went from
    /// ~1.1 s to 69 s.
    #[test]
    fn apparent_reuse_is_still_charged_against_the_ceiling() {
        let want = 16 * MIB;
        let used = 13558 * MIB;
        let reserved = 14656 * MIB;
        let os_needed = want.saturating_sub(reserved - used); // 0 by the estimate
        assert_eq!(os_needed, 0, "the aggregate gap appears to cover this");

        let (budget_ok, free_ok) =
            gate_decide(want, used, os_needed, 338 * MIB, 15062 * MIB, 1492 * MIB);

        // used + want + reserve = 15066 > ceiling 15062. The estimate cannot be
        // trusted to mean "no new OS memory", so the ceiling still binds.
        assert!(
            !budget_ok,
            "the ceiling must bind even when reuse looks free"
        );
        // `free_ok` legitimately passes: it only ever gates measured growth.
        assert!(free_ok);
    }

    /// Growth is still gated by the ceiling: the same overshoot refuses when the
    /// allocation actually has to take memory from the OS.
    #[test]
    fn growth_past_the_ceiling_is_still_refused() {
        let want = 16 * MIB;
        let used = 13558 * MIB;
        let reserved = used; // no gap at all → every byte is new OS memory
        let os_needed = want.saturating_sub(reserved - used);
        assert_eq!(os_needed, want);

        let (budget_ok, _) =
            gate_decide(want, used, os_needed, 4096 * MIB, 15062 * MIB, 1492 * MIB);
        assert!(!budget_ok, "growth past the ceiling must still be refused");
    }

    /// A partial reuse — the gap covers only some of the request — is treated as
    /// growth for the remainder, so the reuse path can't be used to smuggle an
    /// unbounded allocation past the gate.
    #[test]
    fn partial_reuse_is_gated_on_the_growing_remainder() {
        let want = 100 * MIB;
        let used = 1000 * MIB;
        let reserved = 1040 * MIB; // 40 MiB gap → 60 MiB must come from the OS
        let os_needed = want.saturating_sub(reserved - used);
        assert_eq!(os_needed, 60 * MIB);

        // Driver-free is below reserve + the growing remainder → refused.
        let (_, free_ok) = gate_decide(want, used, os_needed, 100 * MIB, 8000 * MIB, 512 * MIB);
        assert!(
            !free_ok,
            "the growing remainder is still floored on driver-free"
        );
    }

    /// Growth that fits keeps passing — the fix must not turn the gate off.
    #[test]
    fn growth_that_fits_is_permitted() {
        let (budget_ok, free_ok) = gate_decide(
            16 * MIB,
            1000 * MIB,
            16 * MIB,
            4096 * MIB,
            15062 * MIB,
            1492 * MIB,
        );
        assert!(budget_ok && free_ok);
    }
}
