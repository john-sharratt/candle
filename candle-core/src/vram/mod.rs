//! The VRAM Governor — a single, cross-platform authority over GPU VRAM
//! residency for the inference engine.
//!
//! One principle drives the whole module: the **real free-VRAM measurement is
//! the single source of truth**. We do not keep a `ceiling − Σcommitted` tally
//! that can drift; we measure, classify each allocation by [`AllocClass`], and
//! size the permanent claims a model load makes from what the measurement says
//! is left. See `docs/vram_governor_design.md` for the full design.
//!
//! The governor's role is **startup**: balloon to find resident capacity `C`,
//! size the expert cache and the KV reservation from it, then freeze the
//! partition. It had a runtime one too — a criticality ladder of registered
//! relief closures, escalated rung by rung whenever a live measurement said the
//! card was tight. That went with the KV cache's move onto a static reservation:
//! there is nothing left to regulate at runtime, because the memory is claimed
//! once and the scheduler reads an exact free-region count rather than asking
//! the governor what the card can spare (`docs/archived/arena_unification.md` §3.8, §5).
//!
//! Layout (one concern per file):
//! - [`reading`] — [`VramReading`], the [`VramProbe`] trait, the test double.
//! - [`budget`] — [`GovernorConfig`] and the KV floor.
//! - [`managed`] — the capacity arithmetic the startup partition is sized from.
//! - [`diag`] — the [`BudgetTable`] snapshot and logging.
//! - [`balloon`] — the balloon-and-measure bootstrap.
//! - `probe_cuda` / `probe_dxgi` — the real measurement backends.

pub mod balloon;
mod budget;
mod diag;
mod host_probe;
mod managed;
pub mod reading;

#[cfg(feature = "cuda")]
mod probe_cuda;
#[cfg(all(windows, feature = "cuda"))]
mod probe_dxgi;

pub use budget::GovernorConfig;
pub use diag::{BudgetRow, BudgetTable};
pub use host_probe::{
    available_low_water, available_physical_ram, host_perf, host_ram_budget, host_ram_budget_from,
    launch_available_ram, pages_in_per_sec, sample_available_low_water, snapshot_launch,
    total_physical_ram, HostPerf, HostRamBudget, PAGEABLE_RESERVE,
};
pub use managed::is_oom;
#[cfg(all(windows, feature = "cuda"))]
pub use probe_dxgi::DxgiProbe;
pub use reading::{BudgetWatchHandle, ProbeKind, VramProbe, VramReading};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::{Device, Result};
use balloon::BalloonAllocator;
use reading::VramReading as Reading;

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
    /// Returns the pool's reusable-but-reserved bytes (`pool_reserved −
    /// pool_used`) — memory a new allocation reuses with no new OS allocation, so
    /// it counts toward the true available even though DXGI headroom (which tracks
    /// `reserved`) can't see it. No-op (0) for a probe-only / test governor.
    config: GovernorConfig,
    capacity_c: AtomicU64,
    /// Probe headroom at the instant `C` was measured — the baseline our own
    /// consumption is counted from.
    ///
    /// `total - headroom` is NOT our usage: DXGI reports
    /// `headroom = Budget - CurrentUsage`, so that difference is
    /// `(total - Budget) + CurrentUsage` and the first term is the OS reserve —
    /// exactly what the balloon already discovered and excluded from `C`.
    /// Subtracting it again double-books it. Measuring the *drop* in headroom
    /// since `C` was taken cancels the reserve out: both readings carry it.
    ///
    /// `0` ⇒ `C` was never measured, so there is no baseline to count from.
    headroom_at_capacity: AtomicU64,
    class_reserved: [AtomicU64; AllocClass::COUNT],
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
            .finish()
    }
}

impl VramGovernor {
    /// Build a governor from a measurement backend and config.
    pub fn new(gpu_id: usize, probe: Box<dyn VramProbe>, config: GovernorConfig) -> Self {
        Self {
            gpu_id,
            probe,
            config,
            capacity_c: AtomicU64::new(0),
            headroom_at_capacity: AtomicU64::new(0),
            class_reserved: Default::default(),
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

    pub fn gpu_id(&self) -> usize {
        self.gpu_id
    }

    /// Take one honest measurement (the source of truth).
    pub fn measure(&self) -> Result<VramReading> {
        self.probe.read()
    }

    fn measure_or_default(&self) -> Reading {
        self.probe
            .read()
            .unwrap_or_else(|_| VramReading::new(0, 0, ProbeKind::Fake))
    }

    /// Directly set the measured capacity `C` (test hook / after an external
    /// balloon). Normal path is [`Self::run_balloon`].
    pub fn set_capacity(&self, c: u64) {
        self.record_capacity(c);
    }

    /// Store `C` together with the headroom reading it was measured against, so
    /// [`Self::expert_budget`] can count our own consumption from that baseline
    /// (see [`Self::headroom_at_capacity`]). Every path that establishes `C`
    /// goes through here — a `C` without its baseline would silently fall back
    /// to the headroom-only budget.
    fn record_capacity(&self, c: u64) {
        self.capacity_c.store(c, Ordering::Relaxed);
        let headroom = self.probe.read().map(|r| r.headroom).unwrap_or(0);
        self.headroom_at_capacity.store(headroom, Ordering::Relaxed);
    }

    /// Headroom at the moment `C` was measured; `0` when `C` was never set.
    pub(crate) fn headroom_at_capacity(&self) -> u64 {
        self.headroom_at_capacity.load(Ordering::Relaxed)
    }

    /// Run the balloon through `alloc`, record the resident high-water as `C`,
    /// and return it. Applies the circuit-breaker fallback if the claim is
    /// implausibly small.
    ///
    /// Fast path: if the measurement already reports headroom at/above the
    /// capacity target, the card is already ours — there are no squatters to
    /// evict, so the live measurement *is* the capacity and the expensive
    /// touch-balloon is skipped. This is the normal startup case (empty card
    /// before model load); the touch only runs when VRAM is genuinely contended.
    ///
    /// **The reserve is applied here too.** It was not: this path used the
    /// fraction as a *threshold* and then took `headroom.min(total)`, so on the
    /// path that actually runs in production the card kept nothing back at all.
    /// Both paths now clamp to the same [`balloon::capacity_target`].
    pub fn run_balloon(&self, alloc: &mut dyn BalloonAllocator) -> Result<u64> {
        let reading = self.probe.read()?;
        let target = balloon::capacity_target(reading.total, self.config.capacity_reserve);
        // The same wobble margin the growth loop applies (see
        // `balloon::wobble_margin`): the fast path may only skip the balloon
        // when the headroom clears the target WITH that slack, otherwise a
        // WDDM system whose idle budget momentarily exceeds `total − reserve`
        // would take the uncapped target here and land `C` in demotion
        // territory. On non-WDDM probes the margin is the reserve itself and
        // the condition reduces to the old `headroom ≥ target` (target already
        // holds the reserve back).
        let margin = balloon::wobble_margin(&reading, &self.config);
        if reading.headroom.saturating_sub(margin) >= target {
            let c = reading.headroom.min(target);
            self.record_capacity(c);
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
        // Half the card. A claim below this is not a tight card, it is a broken
        // measurement — the circuit breaker exists for a probe that reports
        // nonsense, not for a card that is merely busy.
        let sane_floor = total / 2;
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
        self.record_capacity(c);
        Ok(c)
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
        Ok(Arc::new(Self::new(gpu_id, probe, config)))
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

/// What a pinned allocation is *for*.
///
/// The total alone cannot answer the question a pinned-memory decision actually
/// asks — "the tier is short, who else is holding the pages?" — because every
/// consumer here is claiming from the same half-of-RAM ceiling
/// ([`host_probe::PINNABLE_FRACTION`]). One 11 GB claimant and six small ones
/// need telling apart before any of them is resized.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PinnedUse {
    /// A model's weight warm tier — the dominant claimant by an order of
    /// magnitude, and the one whose size is a tuning decision.
    ///
    /// A routed checkpoint's is its experts, a dense one's is its layers, and a
    /// model is one or the other — so they share the name as they share the
    /// sizing arithmetic (`expert_lre::handle::warm_slots_for`).
    WeightWarmTier,
    /// Staging buffers for host↔device transfers that are not the warm tier:
    /// the cold pack's read ring, the KV migration scratch.
    Staging,
    /// Host-mapped allocations the GPU reads in place — the embedding table's
    /// quantized rows, and anything else gathered rather than copied.
    HostMapped,
    /// Device-visible descriptor rings: the kernel dispatch table ring.
    DispatchTables,
}

impl PinnedUse {
    /// Every variant, for reporting.
    pub const ALL: [PinnedUse; 4] = [
        PinnedUse::WeightWarmTier,
        PinnedUse::Staging,
        PinnedUse::HostMapped,
        PinnedUse::DispatchTables,
    ];

    /// A short label for a report line.
    pub fn label(self) -> &'static str {
        match self {
            PinnedUse::WeightWarmTier => "weight warm tier",
            PinnedUse::Staging => "staging buffers",
            PinnedUse::HostMapped => "host-mapped weights",
            PinnedUse::DispatchTables => "dispatch tables",
        }
    }

    fn slot(self) -> &'static AtomicU64 {
        match self {
            PinnedUse::WeightWarmTier => &PINNED_BY_USE[0],
            PinnedUse::Staging => &PINNED_BY_USE[1],
            PinnedUse::HostMapped => &PINNED_BY_USE[2],
            PinnedUse::DispatchTables => &PINNED_BY_USE[3],
        }
    }
}

static PINNED_BY_USE: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Subtract `bytes` from `at`, saturating — pinned frees can outrun their
/// allocs across the two gauges during teardown, and a wrap would read as
/// terabytes held.
fn saturating_sub_at(at: &AtomicU64, bytes: u64) {
    let mut cur = at.load(Ordering::Relaxed);
    loop {
        let next = cur.saturating_sub(bytes);
        match at.compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => cur = observed,
        }
    }
}

/// Record `bytes` of newly allocated host-pinned memory, attributed to `use_`.
pub fn note_host_pinned_alloc(use_: PinnedUse, bytes: u64) {
    HOST_PINNED_BYTES.fetch_add(bytes, Ordering::Relaxed);
    use_.slot().fetch_add(bytes, Ordering::Relaxed);
}

/// Record `bytes` of host-pinned memory returned to the OS.
pub fn note_host_pinned_free(use_: PinnedUse, bytes: u64) {
    saturating_sub_at(&HOST_PINNED_BYTES, bytes);
    saturating_sub_at(use_.slot(), bytes);
}

/// Total host-pinned bytes currently allocated process-wide.
pub fn host_pinned_bytes() -> u64 {
    HOST_PINNED_BYTES.load(Ordering::Relaxed)
}

/// Host-pinned bytes held by one consumer.
pub fn host_pinned_bytes_for(use_: PinnedUse) -> u64 {
    use_.slot().load(Ordering::Relaxed)
}

/// Every consumer's current holding, for a report. Sums to
/// [`host_pinned_bytes`].
pub fn host_pinned_breakdown() -> [(PinnedUse, u64); 4] {
    PinnedUse::ALL.map(|u| (u, host_pinned_bytes_for(u)))
}

/// Return the CUDA async pool's reserved-but-unused memory to the OS, once
/// loading is done. `(reserved_before, reserved_after)`, or `None` off CUDA.
///
/// The pool reserves its peak from the OS and by design never gives it back, so
/// whatever it holds sits for the life of the daemon OUTSIDE the KV reservation
/// — and it is the first thing WDDM spills to host RAM when the card fills.
/// Measured on the 3.6-35B under load: 4.75 GiB reserved against 3.03 GiB live,
/// and the 1.72 GiB gap matched what the driver had demoted almost exactly.
///
/// **On that model this reclaims nothing, and the reason is worth keeping.**
/// Loading is *not* where the pool peaks: right after the weights and experts
/// are staged it holds 1,952 MiB with no free blocks at all. The gap opens
/// later, as wave width grows during calibration and ingest, which is exactly
/// where `cuMemPoolTrimTo`'s synchronous unmap is unsafe. So this is the right
/// reclaim at the only safe moment, and for a workload whose pool peaks at
/// runtime it is a no-op — the fix for that one is to stop routing large
/// runtime allocations through this pool, not to trim it harder.
///
/// **Call this only between load and serving.** `cuMemPoolTrimTo` unmaps
/// synchronously rather than stream-ordered, so it needs a moment when no
/// kernel holds a pointer into the freed blocks; the synchronize below makes
/// that true rather than assumed. Only blocks nothing is using are released —
/// live allocations, weights and staged experts included, are untouched.
///
/// Lives here rather than in the caller because `feature = "cuda"` is a
/// candle-core feature: a `#[cfg(feature = "cuda")]` written in a crate that
/// has no such feature is not a guard, it is an unconditional deletion of the
/// code it wraps.
pub fn trim_pool_after_load(device: &Device) -> Option<(usize, usize)> {
    #[cfg(feature = "cuda")]
    {
        let Device::Cuda(cuda) = device else {
            return None;
        };
        let before = cuda.pool_reserved_bytes().ok()?;
        if let Err(e) = device.synchronize() {
            tracing::warn!(
                target: "candle_core::vram",
                "post-load pool trim: device sync failed ({e}); leaving the pool alone"
            );
            return None;
        }
        if let Err(e) = cuda.trim_pool(0) {
            tracing::warn!(target: "candle_core::vram", "post-load pool trim failed: {e}");
            return None;
        }
        let after = cuda.pool_reserved_bytes().unwrap_or(before);
        Some((before, after))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        None
    }
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
