//! The VRAM Governor — a single, cross-platform authority over GPU VRAM
//! residency for the inference engine.
//!
//! One principle drives the whole module: the **real free-VRAM measurement is
//! the single source of truth**. We do not keep a `ceiling − Σcommitted` tally
//! that can drift; we measure, classify each allocation by [`AllocClass`], let
//! the budget evolve, and relieve pressure cheapest-first on a criticality
//! ladder — syncing the GPU only at the top rung. See
//! `docs/vram_governor_design.md` for the full design.
//!
//! Layout (one concern per file):
//! - [`reading`] — [`VramReading`], the [`VramProbe`] trait, the test double.
//! - [`budget`] — [`GovernorConfig`], the KV floor and ladder thresholds.
//! - [`relief`] — the criticality registry and the escalating relief loop.
//! - [`managed`] — [`VramGovernor::reserve`]/[`VramGovernor::allocate`]/forecast.
//! - [`diag`] — the [`BudgetTable`] snapshot and logging.
//! - [`balloon`] — the balloon-and-measure bootstrap.
//! - `probe_cuda` / `probe_dxgi` — the real measurement backends.

pub mod balloon;
mod budget;
mod diag;
mod host_probe;
mod managed;
pub mod reading;
mod relief;

#[cfg(feature = "cuda")]
mod probe_cuda;
#[cfg(all(windows, feature = "cuda"))]
mod probe_dxgi;

pub use budget::{GovernorConfig, LadderTier};
pub use diag::{BudgetRow, BudgetTable};
pub use host_probe::{
    host_perf, host_ram_budget, host_ram_budget_from, pages_in_per_sec, HostPerf, HostRamBudget,
};
pub use managed::is_oom;
pub use reading::{BudgetWatchHandle, ProbeKind, VramProbe, VramReading};
pub use relief::{KvReliefDriver, ReliefHandle, ReliefOutcome, ReliefRequest, ReliefResult};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;

use crate::Result;
use balloon::BalloonAllocator;
use reading::VramReading as Reading;
use relief::ReliefRegistry;

/// What an allocation is *for*. Drives evictability (which relief rung, if any,
/// may reclaim it), the concurrency forecast, and the budget table — **never**
/// summed as an availability gate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AllocClass {
    /// Mandatory model tensors — permanent, never evicted.
    Weights,
    /// MoE expert slots — a fixed pool; evictable only at `Critical`.
    Expert,
    /// Forward intermediates + grow-once scratch pools — needed, not evicted.
    Scratch,
    /// KV cache — the variable, evictable region.
    Kv,
}

impl AllocClass {
    pub const COUNT: usize = 4;
    pub const ALL: [AllocClass; Self::COUNT] = [
        AllocClass::Weights,
        AllocClass::Expert,
        AllocClass::Scratch,
        AllocClass::Kv,
    ];
    pub fn idx(self) -> usize {
        match self {
            AllocClass::Weights => 0,
            AllocClass::Expert => 1,
            AllocClass::Scratch => 2,
            AllocClass::Kv => 3,
        }
    }
}

/// Relief rungs in ascending *future penalty* (cheapest-to-recover-from first).
/// Only `Critical` synchronises the GPU (see `docs/vram_governor_design.md` §8).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Criticality {
    /// Free/near-free, no data movement (release empty arenas, trim settled pool).
    Trivial,
    /// Lossless on-GPU reshuffle (compact / defrag).
    Cheap,
    /// Compress-to-free: quantize COMPLETED float turns early. A net shrink, not
    /// a move — the turn stays resident and attended-over, and it is *no extra
    /// loss* (completed turns are quantized on seal regardless; pressure only
    /// pulls it forward). That zero incremental penalty is why it precedes the
    /// reversible-but-reload-costing eviction below.
    Moderate,
    /// Reversible tier demotion that MOVES data off the card (hot→warm evict,
    /// slot→pinned) — reloaded if re-attended, hence a higher future penalty
    /// than compressing turns that stay resident.
    Costly,
    /// Aggressive: GPU sync + remeasure, drop-to-cold, expert-pool shrink.
    Critical,
}

impl Criticality {
    pub const ALL: [Criticality; 5] = [
        Criticality::Trivial,
        Criticality::Cheap,
        Criticality::Moderate,
        Criticality::Costly,
        Criticality::Critical,
    ];
    pub fn idx(self) -> usize {
        match self {
            Criticality::Trivial => 0,
            Criticality::Cheap => 1,
            Criticality::Moderate => 2,
            Criticality::Costly => 3,
            Criticality::Critical => 4,
        }
    }
}

/// Fallback headroom kept free from capacity when the balloon can't establish a
/// trustworthy claim (circuit breaker).
const BALLOON_FALLBACK_MARGIN: u64 = 1024 * 1024 * 1024;

/// The per-GPU authority. Constructed with a [`VramProbe`] (the measurement
/// backend) and an optional GPU-sync hook (invoked only at the `Critical` rung).
pub struct VramGovernor {
    gpu_id: usize,
    probe: Box<dyn VramProbe>,
    /// Called at the `Critical` rung to retire pending async frees before the
    /// ground-truth remeasure. No-op for a probe-only (test / CPU) governor.
    sync_hook: Box<dyn Fn() + Send + Sync>,
    /// Returns the pool's reusable-but-reserved bytes (`pool_reserved −
    /// pool_used`) — memory a new allocation reuses with no new OS allocation, so
    /// it counts toward the true available even though DXGI headroom (which tracks
    /// `reserved`) can't see it. No-op (0) for a probe-only / test governor.
    reuse_hook: Box<dyn Fn() -> u64 + Send + Sync>,
    sync_calls: AtomicU64,
    config: GovernorConfig,
    capacity_c: AtomicU64,
    class_reserved: [AtomicU64; AllocClass::COUNT],
    relief: RwLock<ReliefRegistry>,
    last_relief: Mutex<Option<(Criticality, u64)>>,
    last_critical: Mutex<Option<Instant>>,
    /// Count of background-allocator VRAM-exhaustion reports since the scheduler
    /// last drained it (see [`Self::signal_starvation`]). Lets a starved
    /// background compressor ask the scheduler for an escalated recovery.
    starvation: AtomicU64,
}

impl std::fmt::Debug for VramGovernor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VramGovernor")
            .field("gpu_id", &self.gpu_id)
            .field("capacity_c", &self.capacity())
            .field("relief_count", &self.relief_count())
            .finish()
    }
}

impl VramGovernor {
    /// Build a governor from a measurement backend and config. The GPU-sync hook
    /// defaults to a no-op (correct for tests and CPU); wire a real one with
    /// [`Self::with_sync_hook`] or construct via [`Self::from_device`].
    pub fn new(gpu_id: usize, probe: Box<dyn VramProbe>, config: GovernorConfig) -> Self {
        Self {
            gpu_id,
            probe,
            sync_hook: Box::new(|| {}),
            reuse_hook: Box::new(|| 0),
            sync_calls: AtomicU64::new(0),
            config,
            capacity_c: AtomicU64::new(0),
            class_reserved: Default::default(),
            relief: RwLock::new(ReliefRegistry::default()),
            last_relief: Mutex::new(None),
            last_critical: Mutex::new(None),
            starvation: AtomicU64::new(0),
        }
    }

    /// Report that a background allocator (e.g. the persistence hot→warm
    /// compress-to-free pass) FAILED to allocate because VRAM is exhausted. This
    /// is the strongest "critically full" signal the system gets — the caller
    /// couldn't even fit a small transient arena. The failing operation stays
    /// non-destructive (its data is untouched, it retries later); this only asks
    /// the owner of relief (the scheduler) to make room for that retry. Cheap and
    /// thread-safe: any thread may call it. Drained via [`Self::take_starvation`].
    pub fn signal_starvation(&self) {
        self.starvation.fetch_add(1, Ordering::Relaxed);
    }

    /// Take and clear the starvation count reported since the last call. Non-zero
    /// ⇒ a background allocator hit VRAM exhaustion and the caller should escalate
    /// relief (evict deeper) so its retry succeeds. Counting (not a bool) so a
    /// burst is visible in the escalation log.
    pub fn take_starvation(&self) -> u64 {
        self.starvation.swap(0, Ordering::Relaxed)
    }

    /// Install the GPU-sync hook invoked at the `Critical` rung (e.g.
    /// `device.synchronize()`), retiring pending async frees before remeasure.
    pub fn with_sync_hook(mut self, hook: Box<dyn Fn() + Send + Sync>) -> Self {
        self.sync_hook = hook;
        self
    }

    pub fn gpu_id(&self) -> usize {
        self.gpu_id
    }

    /// Take one honest measurement (the source of truth).
    pub fn measure(&self) -> Result<VramReading> {
        self.probe.read()
    }

    /// Install the reuse hook (pool `reserved − used`) so [`Self::available`] and
    /// the relief ladder count the reusable pool free-list.
    pub fn with_reuse_hook(mut self, hook: Box<dyn Fn() -> u64 + Send + Sync>) -> Self {
        self.reuse_hook = hook;
        self
    }

    /// Bytes allocatable **without paging**: honest free headroom PLUS the
    /// reusable pool free-list. This — not raw DXGI headroom — is what the relief
    /// ladder gates on, so eviction (which grows the reuse pool but doesn't lower
    /// `reserved`, hence doesn't move DXGI headroom) is correctly seen as relief.
    pub fn available(&self) -> Result<u64> {
        Ok(self.measure()?.headroom.saturating_add((self.reuse_hook)()))
    }

    fn measure_or_default(&self) -> Reading {
        self.probe
            .read()
            .unwrap_or_else(|_| VramReading::new(0, 0, ProbeKind::Fake))
    }

    /// Invoked only at the `Critical` rung: count it and run the sync hook.
    pub(crate) fn do_sync(&self) {
        self.sync_calls.fetch_add(1, Ordering::Relaxed);
        (self.sync_hook)();
    }

    /// Directly set the measured capacity `C` (test hook / after an external
    /// balloon). Normal path is [`Self::run_balloon`].
    pub fn set_capacity(&self, c: u64) {
        self.capacity_c.store(c, Ordering::Relaxed);
    }

    /// Run the balloon through `alloc`, record the resident high-water as `C`,
    /// and return it. Applies the circuit-breaker fallback if the claim is
    /// implausibly small.
    ///
    /// Fast path: if the measurement already reports headroom at/above the balloon
    /// target, the card is already ours — there are no squatters to evict, so the
    /// live measurement *is* the capacity and the expensive touch-balloon is
    /// skipped. This is the normal startup case (empty card before model load);
    /// the touch only runs when VRAM is genuinely contended.
    pub fn run_balloon(&self, alloc: &mut dyn BalloonAllocator) -> Result<u64> {
        let reading = self.probe.read()?;
        let target = (self.config.balloon_target_frac * reading.total as f64) as u64;
        if reading.headroom >= target {
            let c = reading.headroom.min(reading.total);
            self.capacity_c.store(c, Ordering::Relaxed);
            tracing::info!(
                target: "candle_core::vram",
                "balloon skipped: card already free (headroom {}MiB ≥ target {}MiB) — C={}MiB",
                reading.headroom / (1024 * 1024),
                target / (1024 * 1024),
                c / (1024 * 1024),
            );
            return Ok(c);
        }
        self.run_full_balloon(alloc)
    }

    /// Always run the touch-balloon (no fast-path skip), record `C`, and return
    /// it. Used when VRAM is contended, or to force a claim for diagnostics.
    pub fn run_full_balloon(&self, alloc: &mut dyn BalloonAllocator) -> Result<u64> {
        let claimed = balloon::balloon_measure(self.probe.as_ref(), alloc, &self.config)?;
        let total = self.probe.read()?.total;
        let sane_floor = (self.config.balloon_target_frac * 0.5 * total as f64) as u64;
        let c = if claimed >= sane_floor {
            claimed
        } else {
            tracing::warn!(
                target: "candle_core::vram",
                "balloon claimed only {}MiB of {}MiB total — falling back to total − margin",
                claimed / (1024 * 1024),
                total / (1024 * 1024),
            );
            total.saturating_sub(BALLOON_FALLBACK_MARGIN)
        };
        self.capacity_c.store(c, Ordering::Relaxed);
        Ok(c)
    }

    /// Spawn the reactive budget watcher: if the probe exposes an OS
    /// budget-change event (Windows), block on it and run threshold-gated relief
    /// on each signal so we shed KV the instant another process steals VRAM,
    /// before the OS pages us. No-op where no push event exists.
    pub fn spawn_budget_watcher(governor: Arc<VramGovernor>) {
        let Some(watch) = governor.probe.budget_change_event() else {
            return;
        };
        std::thread::Builder::new()
            .name("vram-budget-watch".into())
            .spawn(move || loop {
                // Wake at least once a second even absent a signal, as a poll.
                let _fired = watch.wait(1000);
                if let Err(e) = governor.relieve_pressure(AllocClass::Kv) {
                    tracing::debug!(target: "candle_core::vram", "budget-watch relief error: {e}");
                }
            })
            .ok();
    }
}

// ── Real device wiring (CUDA) ────────────────────────────────────────────────

#[cfg(feature = "cuda")]
impl VramGovernor {
    /// Construct a governor for a real CUDA device: pick the best probe (DXGI on
    /// Windows matched by LUID, else `cuMemGetInfo`) and wire the sync hook to the
    /// device stream. Returns an `Arc` ready to [`install`].
    pub fn from_device(device: &crate::Device, gpu_id: usize) -> Result<Arc<Self>> {
        Self::from_device_with_config(device, gpu_id, GovernorConfig::default())
    }

    /// As [`Self::from_device`] but with an explicit config (tests / tuning).
    pub fn from_device_with_config(
        device: &crate::Device,
        gpu_id: usize,
        config: GovernorConfig,
    ) -> Result<Arc<Self>> {
        let probe: Box<dyn VramProbe> = Self::pick_probe(device, gpu_id)?;
        let sync_dev = device.clone();
        // The Critical rung retires pending async frees AND trims the pool, so the
        // freed bytes actually return to the OS before the ground-truth remeasure
        // (the async pool otherwise retains them and the remeasure sees no gain).
        let reuse_dev = device.clone();
        let gov = Self::new(gpu_id, probe, config)
            .with_sync_hook(Box::new(move || {
                // Trim under the registered arena-topology guard: `trim_pool(0)`
                // (`cuMemPoolTrimTo`) synchronously unmaps freed pool memory, which
                // would fault an in-flight hot→warm migrate's captured base pointers.
                // Skipped (freed bytes stay pooled, returned on a later pass) while a
                // migrate holds the topology. No guard registered → runs directly.
                guarded_pool_trim(|| {
                    let _ = sync_dev.synchronize();
                    if let crate::Device::Cuda(d) = &sync_dev {
                        let _ = d.trim_pool(0);
                    }
                });
            }))
            // Reusable pool free-list: freed KV memory the pool holds but a new
            // allocation reuses with no new OS allocation (§ vram_budget_available).
            .with_reuse_hook(Box::new(move || {
                if let crate::Device::Cuda(d) = &reuse_dev {
                    let r = d.pool_reserved_bytes().unwrap_or(0);
                    let u = d.pool_used_bytes().unwrap_or(0);
                    r.saturating_sub(u) as u64
                } else {
                    0
                }
            }));
        Ok(Arc::new(gov))
    }

    fn pick_probe(device: &crate::Device, _gpu_id: usize) -> Result<Box<dyn VramProbe>> {
        #[cfg(windows)]
        {
            if let crate::Device::Cuda(cuda) = device {
                match probe_dxgi::DxgiProbe::for_cuda_device(cuda) {
                    Ok(p) => return Ok(Box::new(p)),
                    Err(e) => tracing::warn!(
                        target: "candle_core::vram",
                        "DXGI probe unavailable ({e}); falling back to cuMemGetInfo"
                    ),
                }
            }
        }
        Ok(Box::new(probe_cuda::CudaProbe::new(device.clone())))
    }
}

// ── Process-global registry (one governor per physical GPU) ───────────────────

static REGISTRY: OnceLock<Mutex<HashMap<usize, Arc<VramGovernor>>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<usize, Arc<VramGovernor>>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Install (or replace) the governor for its GPU so KV and expert code can reach
/// the same instance via [`get`].
pub fn install(governor: Arc<VramGovernor>) {
    registry().lock().unwrap().insert(governor.gpu_id, governor);
}

/// The governor for `gpu_id`, if installed.
pub fn get(gpu_id: usize) -> Option<Arc<VramGovernor>> {
    registry().lock().unwrap().get(&gpu_id).cloned()
}

/// A process-global guard wrapper for the async-pool trim. `cuMemPoolTrimTo`
/// (via [`crate::CudaDevice::trim_pool`]) **synchronously unmaps** freed pool
/// memory process-wide, which is unsafe while another subsystem has captured raw
/// device base pointers with no lock held — candle-nn's hot→warm KV migrate
/// builds a dense per-arena base-pointer table and launches kernels that
/// dereference it, unlocked, on the persistence thread. candle-nn's own trim
/// paths hold its arena-topology relief guard, but the governor's Critical-rung
/// sync-hook lives HERE in candle-core — below candle-nn — and cannot reach that
/// guard. So the hot→warm layer registers a wrapper that acquires the guard
/// *around* the trim; unregistered (tests / non-KV use), the trim runs directly.
///
/// The wrapper receives the trim closure and decides whether / when to run it,
/// holding the guard for the trim's whole duration. (A bare "is a migrate in
/// flight?" pre-check is a TOCTOU race — a migrate can start right after the
/// check; safety needs the guard held across the trim.)
type PoolTrimGuard = Box<dyn Fn(&mut dyn FnMut()) + Send + Sync>;
static POOL_TRIM_GUARD: OnceLock<PoolTrimGuard> = OnceLock::new();

/// Register the arena-topology guard wrapper for the governor's pool trim (see
/// [`PoolTrimGuard`]). Idempotent — the first registration wins.
pub fn set_pool_trim_guard(guard: PoolTrimGuard) {
    let _ = POOL_TRIM_GUARD.set(guard);
}

/// Run `trim` under the registered pool-trim guard, or directly if none is set.
#[allow(dead_code)] // the only caller (the governor sync-hook trim) degenerates on non-CUDA builds
fn guarded_pool_trim(mut trim: impl FnMut()) {
    match POOL_TRIM_GUARD.get() {
        Some(g) => g(&mut trim),
        None => trim(),
    }
}

/// Remove the governor for `gpu_id` (test cleanup).
pub fn remove(gpu_id: usize) {
    registry().lock().unwrap().remove(&gpu_id);
}

// ── Host-pinned gauge ────────────────────────────────────────────────────────
// Process-wide tally of host RAM held in NON-PAGEABLE pinned allocations
// (`cuMemAllocHost` pools — the MoE expert host tier is the dominant one, 11 GB
// on the current dev box). Pinned memory is invisible to per-allocation
// accounting downstream but is exactly the bytes a host-RAM availability
// measurement must treat as structural: it can never be paged or reclaimed, so
// "available < floor" while this is large is a permanent condition, not
// pressure. Lives here (not in the allocating crate) because the gauge must be
// readable from CPU-only builds — the memory report is built unconditionally,
// while the allocators are `cfg(cuda)`.

static HOST_PINNED_BYTES: AtomicU64 = AtomicU64::new(0);

/// Record `bytes` of newly allocated host-pinned memory.
pub fn note_host_pinned_alloc(bytes: u64) {
    HOST_PINNED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Record `bytes` of host-pinned memory returned to the OS.
pub fn note_host_pinned_free(bytes: u64) {
    let mut cur = HOST_PINNED_BYTES.load(Ordering::Relaxed);
    loop {
        let next = cur.saturating_sub(bytes);
        match HOST_PINNED_BYTES.compare_exchange_weak(
            cur,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => cur = observed,
        }
    }
}

/// Total host-pinned bytes currently allocated process-wide.
pub fn host_pinned_bytes() -> u64 {
    HOST_PINNED_BYTES.load(Ordering::Relaxed)
}

/// Size of the mmap-backed model weights, recorded once at load.
///
/// The host-RAM budget treats the weights as FULLY RESERVED even though they
/// are file-backed: letting the OS evict weight pages to make room for warm KV
/// trades cheap-tier capacity for hard faults on the inference path, which is
/// the one exchange the budget exists to forbid. (Only when the weights exceed
/// `total − buffer` do they get capped — a machine that cannot hold its model
/// resident must swap by definition, and the budget accepts it explicitly
/// rather than pretending otherwise.)
static WEIGHTS_MMAP_BYTES: AtomicU64 = AtomicU64::new(0);

/// Set the mmap-backed weight size — the WHOLE model's mapped size, not one
/// file's.
///
/// Deliberately a store, not an add: a model swap must replace the old figure
/// rather than accumulate across loads. The corollary is that a multi-part
/// loader has to sum its shards and call this **once**. Calling it per shard
/// would leave the gauge holding only the last one, and
/// [`host_ram_budget`](super::vram::host_ram_budget) then under-reserves the
/// weights in exact proportion and hands the shortfall to warm KV as budget it
/// does not have — silently, surfacing only as evicted weight pages and hard
/// faults on the inference path. Named `set_` rather than `note_` (the pinned
/// gauges above accumulate) so a per-shard call reads as wrong where it is
/// written.
pub fn set_weights_mmap(bytes: u64) {
    WEIGHTS_MMAP_BYTES.store(bytes, Ordering::Relaxed);
}

/// The recorded mmap-backed weight size, 0 before any model load.
pub fn weights_mmap_bytes() -> u64 {
    WEIGHTS_MMAP_BYTES.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests;
