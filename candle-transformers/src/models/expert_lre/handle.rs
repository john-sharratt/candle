//! Public API — the [`ExpertCache`] handle and its two operating modes.
//!
//! [`ExpertCache`] is the main entry point for the expert pipeline.  It
//! provides two modes:
//!
//! - **Threaded** (mmap path): a background thread owns all mutable state.
//!   The forward path submits work via a channel and blocks for the result.
//! - **Inline** (reader path): all experts are pre-loaded to VRAM.  No
//!   thread, no DMA.  A Mutex provides interior mutability (uncontended).

use super::cache::{minimum_resident_slots, pinned_layer_count, ExpertCacheInner};
#[cfg(not(feature = "cuda"))]
use super::compute::compute_expert_contribution_gpu_weights;
#[cfg(feature = "cuda")]
use super::compute::compute_experts_grouped;
#[cfg(feature = "cuda")]
use super::gpu_dispatch::GpuDispatchTables;
#[cfg(feature = "cuda")]
use super::pack::{
    open_or_create, repack_fingerprint, LayerSpansInput, PackIdentity, PackSource, PackSpec,
    RecordLayout,
};
#[cfg(feature = "cuda")]
use super::pinned::{stratified_membership, ExpertResidency, WarmPool};
#[cfg(not(feature = "cuda"))]
use super::pipeline::prewarm_expert_cache;
#[cfg(feature = "cuda")]
use super::pipeline::{
    slot_bytes_for, slot_offsets, startup_from_pack, startup_repack, ColdStaging, StartupTargets,
    COLD_STAGING_BUFFERS,
};
use super::pipeline::{spawn_pipeline_thread, PipelineState};
use super::zone_geometry::ZoneGeometry;
use super::transition::TransitionMatrix;
use super::types::{
    ClassifiedExperts, CopyBatchFence, ExpertSlot, MmapExpertRef, MoeInput, MoeWorkRequest,
    PipelineMessage, PipelineStats,
};
use crate::models::profile::{profile_now, ProfileAccumulator, ProfileMark, ProfileSnapshot};

/// Report, once per distinct reason, that a call could not use the device
/// dispatch tables. Once because the caller is a per-layer hot path — the point
/// is that the reason is *stated*, not that it is repeated 40 times a token.
#[cfg(feature = "cuda")]
/// Say once, per *kind* of refusal, why this layer fell off the device path.
///
/// **The key is a `&'static str` and the detail is lazy, and both matter.** This
/// runs per MoE layer per wave. Keying the dedup on the fully formatted message
/// — which is what it used to do — means any refusal carrying a varying number
/// re-logs and inserts another `String` every time that number changes: the
/// zone-retraction refusal interpolates the live capacity and floor, so it would
/// have logged on every distinct geometry and grown the set without bound, on
/// the hot path, while claiming in its own comment to speak once. Formatting the
/// detail behind `FnOnce` also keeps the `format!` off every call that is about
/// to be deduped away.
fn log_dispatch_refusal(reason: &'static str, detail: impl FnOnce() -> String) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if let Ok(mut s) = seen.lock() {
        if s.insert(reason) {
            tracing::warn!(
                reason,
                detail = %detail(),
                "expert cache: this MoE layer takes the host path's blocking routing \
                 readback instead of the device tables"
            );
        }
    }
}
use candle::cuda_backend::wave_provenance::WaveTicket;
use candle::quantized::Int8Mode;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::WeightZone;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaStream;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

// ============================================================================
// Warm tier sizing
// ============================================================================

/// The seed for the warm tier's stratified draw.
///
/// Fixed rather than random so that two runs of the same build on the same model
/// warm the same experts: a change in expert-cache hit rate is then attributable
/// to the change under test, not to which experts got lucky at startup.
const WARM_DRAW_SEED: u64 = 0x5745_524D_5F53_4545;

/// Host RAM left unclaimed by the warm tier, for everything the process needs
/// after it.
///
/// The warm tier is the single largest host allocation the engine makes and the
/// first one it makes, so whatever it takes, the rest of the process must live
/// in what remains — the cold-tier staging ring, the routing buffer, the
/// substrate's pinned cold-load and elevate scratch (192 MiB between them), the
/// `PinnedStager` arenas, and the warm **KV** tier's pageable arenas.
///
/// Sizing the tier to the last free page is what makes this necessary: pinned
/// pages cannot be reclaimed under pressure, so a warm tier that fits by exactly
/// nothing leaves the next allocation to fail instead of merely running slower.
/// That happened — a 46 MB staging ring failed with `CUDA_ERROR_OUT_OF_MEMORY`
/// immediately after a warm pool sized against *total* RAM took every free page
/// on a machine with 12 GB already in use by other processes.
///
/// **4 GiB, and that figure is measured rather than reasoned.** The run's
/// non-pinned transient — everything the process takes after the tier is
/// pinned — is **3.30 GiB** on the 3.6-35B gate: launch 16.45 GiB, tier 12.14,
/// other pinned 0.62, and a free-RAM low-water of 0.39 GiB
/// (`vram::available_low_water`). At the old 3 GiB this was *under*-provisioned:
/// the reserve promised 3 GiB of daylight and delivered 0.39.
///
/// It is far above the ~250 MiB those allocations nominally total because the
/// mapped checkpoint's touched pages dominate them, and because the tier is past
/// its knee well before it runs out of room. Lowering it to 1 GiB was measured
/// on an earlier build: the tier grew from 4,979 slots to 5,090, cold loads
/// halved (986 → 435), and throughput did not improve — flat to 1–2 % down, with
/// single-stream t/s falling further (204.5 → 197.0). Once the draw covers
/// VRAM's complement the remaining cold reads are not the bottleneck, and pinned
/// pages taken past that point come out of the page cache and the warm KV tier,
/// which this gate barely exercises and a daemon workload does. When the
/// performance argument is a wash, the safety argument decides.
pub const WARM_TIER_HEADROOM: u64 = 4 * 1024 * 1024 * 1024;

/// Host RAM that must stay pageable, whatever else happens.
///
/// The page cache, the mapped checkpoint, every other process's working set,
/// and the engine's own non-pinned allocations all live here. Pinning into it
/// does not fail — measured directly: a probe took the entire pinnable half of
/// a 31.5 GiB box without the driver once refusing, and free RAM ended at
/// 0.02 GiB. **There is no natural stopping point**, which is why this bound is
/// stated rather than discovered, and why an allocate-until-refusal probe cannot
/// find it: refusal never comes, the machine just starts thrashing.
/// `candle-core/tests/pinned_ceiling_probe.rs` re-measures it on any machine
/// this needs revisiting on.
///
/// # An absolute floor, not a fraction
///
/// This was `total / 2`, from a measured failure at 76 % on a 194 GB machine
/// (148 GB locked, 66 GB of other commit pushed to pagefile). A fraction reads
/// the right way on that machine and the wrong way on a small one: on a 31.5 GiB
/// box half is 15.76 GiB reserved against an OS and application set measured at
/// 9.5 GiB, so the rule bound the warm tier for no reason anyone could point
/// at — the tier stopped growing while 6 GiB sat unused and unusable.
///
/// What the OS and the surrounding applications need does not scale with how
/// much RAM is installed, so the reserve is an absolute quantity. 10 GiB covers
/// the 9.5 GiB measured here with room, and on a large machine it lets pinning
/// go far past half — which is correct, and is what the 194 GB case was really
/// telling us: 46 GB pageable was too little there too.
const PAGEABLE_RESERVE: u64 = 10 * 1024 * 1024 * 1024;

/// How many warm slots to ask for: **every expert the machine will actually
/// give room for.**
///
/// The target is the whole model. A miss that reaches the cold tier is a
/// synchronous NVMe read on the pipeline thread, so the warm tier is not an
/// optimisation over the pack — it is what keeps the pack off the critical path.
/// The first build of this sized it as a *share* of spare RAM (half), which left
/// 2,241 of 6,144 experts warm and sent **64 % of every miss to disk**;
/// aggregate throughput fell by a third against the two-tier cache it replaced.
///
/// The bound is **available** RAM, not total. Total is what the machine has;
/// available is what it will give, and on a dev box with an editor and a browser
/// open the two differ by 12 GB. `host_ram_budget` reasons in totals because its
/// other callers ask "is this machine big enough for this model", which is a
/// question about the machine. This one is "may I have these pages now", which
/// is a question about this moment.
///
/// `cuMemAllocHost` remains the authority — [`WarmPool::new`] halves on refusal
/// — but a refusal costs half the tier, so the first ask should be one that can
/// succeed.
/// The warm tier's sizing decision, kept so a report can name **which** ceiling
/// bound it.
///
/// Three independent limits compete for the tier (see [`warm_slots_for`]) and
/// they are not close to each other on every machine — on a 32 GB Windows box
/// the pinnable half is the binder, on a 194 GB box it is the weights
/// reservation. Reading three numbers off a log line and inferring the minimum
/// is exactly the step that got skipped when a tier sized at a third of the
/// model went unnoticed while it sent two thirds of every miss to disk.
#[derive(Clone, Copy, Debug, Default)]
pub struct WarmTierSizing {
    pub total_ram: u64,
    /// Free RAM at the instant the tier was sized — mid-load, and reported only
    /// so the gap against `launch_ram` is visible.
    pub available_ram: u64,
    /// Free RAM at process launch: the baseline ceiling 2 is actually built on.
    pub launch_ram: u64,
    /// Host RAM this process had already page-locked when the tier was sized.
    pub already_pinned: u64,
    /// Ceiling 1: what the machine is big enough for once the mmap'd weights
    /// and the OS floor are reserved.
    pub kv_warm_budget: u64,
    /// Ceiling 2: what is free this second, less the headroom the rest of the
    /// process needs after the tier.
    pub available_less_headroom: u64,
    /// Ceiling 3: how much of the machine may be page-locked at all, less what
    /// is already pinned.
    pub pinnable_cap: u64,
    /// The binding ceiling, by name.
    pub bound_by: &'static str,
    /// Bytes the tier actually took, and the slots that bought.
    pub taken_bytes: u64,
    pub slots: usize,
    /// Slots it would have taken with no ceiling at all — one per evictable
    /// expert. The gap against `slots` is the tier's shortfall.
    pub wanted_slots: usize,
    pub stride: usize,
}

/// The three ceilings, named once so [`WarmTierSizing::bound_by`] and any
/// report of it agree by construction rather than by matching prose.
pub const CEILING_HOST_BUDGET: &str = "host RAM budget (weights + OS floor)";
pub const CEILING_AVAILABLE: &str = "available RAM less headroom";
pub const CEILING_PINNABLE: &str = "pinnable region (total less pageable reserve)";
pub const CEILING_NONE: &str = "nothing — the tier holds every evictable expert";

static WARM_SIZING: Mutex<Option<WarmTierSizing>> = Mutex::new(None);

/// The warm tier's sizing decision from this process's model load, if one has
/// happened.
pub fn last_warm_tier_sizing() -> Option<WarmTierSizing> {
    WARM_SIZING.lock().ok().and_then(|g| *g)
}

/// The sizing arithmetic, with every machine reading passed in.
///
/// Split out of [`warm_slots_for`] the way `host_ram_budget_from` is split out
/// of `host_ram_budget`: both machines' numbers, and the shape of every ceiling,
/// pin down in unit tests without touching a process-global gauge or needing a
/// GPU.
///
/// Three ceilings, all real: what the machine is big enough for, what it had
/// free before this process started, and how much of it may be PAGE-LOCKED at
/// all.
///
/// The third is the one the first two cannot see. Availability counts droppable
/// page cache (a 156 GB GGUF mmap reads as "available"), and the warm budget only
/// nets out pinned memory that already exists — so on a model whose experts
/// nearly fill host RAM, both ceilings happily size the tier to the whole expert
/// set. Pinning that much (measured: 148 GB locked of 194 GB, 66 GB of other
/// commit pushed to pagefile) leaves the OS thrashing everything that is not the
/// warm tier. Page-locked memory is capped at HALF the machine: the other half
/// stays pageable for the page cache (which serves the cold pack reads),
/// activations' host shadows, and everything else alive on the box.
#[allow(clippy::too_many_arguments)]
fn warm_sizing_from(
    stride: usize,
    total_experts: usize,
    total_ram: u64,
    available: u64,
    launch_available: u64,
    already_pinned: u64,
    kv_warm_budget: u64,
) -> WarmTierSizing {
    // **The headroom bounds this ceiling too, and that is not cosmetic.**
    //
    // It used to be subtracted only from the availability ceiling, which was
    // safe by accident: availability was measured mid-load and so was always the
    // lowest of the three, and this one never bound. Sizing from the launch
    // baseline raised availability above it for the first time, this ceiling
    // bound, and the tier took the entire pinnable half — 15.14 GiB plus the
    // 0.62 GiB already pinned is exactly half of a 31.5 GiB machine, with no
    // reserve at all. The first forward then died on `CUDA_ERROR_OUT_OF_MEMORY`.
    //
    // A ceiling that can bind has to leave the same room as the ones beside it.
    let pinnable_cap = total_ram
        .saturating_sub(PAGEABLE_RESERVE)
        .saturating_sub(already_pinned)
        .saturating_sub(WARM_TIER_HEADROOM);
    // **What the machine had free before this process started, not what is left
    // now.** The live figure is taken with the checkpoint mapped and being read,
    // so it is depressed by the engine's own transient footprint — and those
    // pages are file-backed and droppable, so they were never this tier's
    // competitors. Measured across one gate run on the 16 GB box: 15.65 GiB free
    // before the process, 12.18 GiB at the moment of this call, over 20 GiB once
    // it exited. Sizing the process's largest and longest-lived allocation from
    // the bottom of that trough cost the tier 3,030 experts.
    //
    // **Both readings are normalised to "free, excluding what we have pinned"
    // before the max.** They are not on the same scale otherwise: the live
    // figure was taken *after* this process page-locked `already_pinned`, so it
    // already excludes those bytes, while the launch figure predates them and
    // does not. Subtracting from whichever won would double-count the pinned
    // bytes on every machine where the live reading is the larger — which is
    // exactly the case this `max` exists to serve, a box that freed RAM since
    // launch.
    let baseline = launch_available
        .saturating_sub(already_pinned)
        .max(available);
    // The headroom is what the rest of the process needs *after* this tier, and
    // stays a constant because it is a guess (see `WARM_TIER_HEADROOM`) rather
    // than a measurement like the term above.
    let available_less_headroom = baseline.saturating_sub(WARM_TIER_HEADROOM);
    let affordable = kv_warm_budget
        .min(available_less_headroom)
        .min(pinnable_cap);
    let slots = if stride == 0 {
        0
    } else {
        ((affordable / stride as u64) as usize).min(total_experts)
    };
    WarmTierSizing {
        total_ram,
        available_ram: available,
        launch_ram: baseline,
        already_pinned,
        kv_warm_budget,
        available_less_headroom,
        pinnable_cap,
        bound_by: if slots == total_experts {
            CEILING_NONE
        } else if affordable == kv_warm_budget {
            CEILING_HOST_BUDGET
        } else if affordable == available_less_headroom {
            CEILING_AVAILABLE
        } else {
            CEILING_PINNABLE
        },
        taken_bytes: (slots * stride) as u64,
        slots,
        wanted_slots: total_experts,
        stride,
    }
}

fn warm_slots_for(stride: usize, total_experts: usize) -> usize {
    if stride == 0 {
        return 0;
    }
    let (Some(total_ram), Some(available)) = (
        candle::vram::total_physical_ram(),
        candle::vram::available_physical_ram(),
    ) else {
        // No probe on this platform: take no warm tier rather than guess at a
        // number that could be most of the machine. Every expert is still
        // served, from the pack.
        tracing::warn!(
            target: "candle_transformers::expert_lre",
            "warm tier: no host-RAM probe on this platform; running cold-tier only"
        );
        return 0;
    };
    let budget = candle::vram::host_ram_budget(total_ram);
    let sizing = warm_sizing_from(
        stride,
        total_experts,
        total_ram,
        available,
        candle::vram::launch_available_ram().unwrap_or(available),
        candle::vram::host_pinned_bytes(),
        budget.kv_warm_budget_bytes,
    );
    let slots = sizing.slots;
    if let Ok(mut g) = WARM_SIZING.lock() {
        *g = Some(sizing);
    }
    tracing::info!(
        target: "candle_transformers::expert_lre",
        total_gib = total_ram as f64 / 1e9,
        available_gib = available as f64 / 1e9,
        budgeted_gib = budget.kv_warm_budget_bytes as f64 / 1e9,
        weights_gib = budget.weights_reserved_bytes as f64 / 1e9,
        take_gib = (slots * stride) as f64 / 1e9,
        slots,
        of = total_experts,
        "warm tier: sized against available RAM"
    );
    slots
}

// ============================================================================
// Pipeline mode
// ============================================================================

/// The two operating modes of the expert cache.
// Exactly one `PipelineMode` exists per `ExpertCache`, and an `ExpertCache` is
// per-model. Boxing `Inline` to even the variants out would buy back a few
// hundred bytes once and pay an indirection on every expert dispatch.
#[allow(clippy::large_enum_variant)]
enum PipelineMode {
    /// Background thread owns all mutable state.  Used for the mmap path
    /// where DMA overlap is active and experts cycle through VRAM.
    Threaded {
        /// Channel to submit work/hints to the pipeline thread.
        tx: mpsc::SyncSender<PipelineMessage>,
    },
    /// All experts pre-loaded to VRAM.  No background thread.  Mutex
    /// protects the inner state (uncontended — layers execute sequentially).
    Inline {
        inner: Mutex<ExpertCacheInner>,
        device: Device,
    },
}

// ============================================================================
// ExpertCache — the public handle
// ============================================================================

/// Global expert cache / pipeline handle.
///
/// Shared across all `SparseMoeBlock`s via `Arc<ExpertCache>`.
///
/// ## Threaded mode (mmap path)
///
/// A background thread owns all mutable state (`ExpertCacheInner`, slots,
/// eviction scores, mmap, copy stream).  The forward path submits a `MoeWorkRequest`
/// via an MPSC channel and blocks on the oneshot response.  The thread
/// does the full classify → DMA → compute loop with `&mut self` — no locks.
///
/// ## Inline mode (reader path)
///
/// All experts are pre-loaded to VRAM.  No DMA, no background thread.
/// A `Mutex<ExpertCacheInner>` provides interior mutability (uncontended).
/// The forward path locks, computes all experts by slot index, releases.
pub struct ExpertCache {
    mode: PipelineMode,
    /// True when all experts fit in VRAM — hint sending is elided.
    all_resident: bool,
    /// The weight zone's live shape, shared with whichever thread owns it.
    ///
    /// The handle cannot reach `ExpertCacheInner` in threaded mode — it is
    /// moved into the pipeline thread's state — so a geometry check written
    /// against `inner` silently does nothing there. This is the same three
    /// numbers, published lock-free, readable in either mode.
    geometry: Arc<ZoneGeometry>,
    /// Dedicated CUDA stream for async routing index DtoH.
    /// Lives on the forward thread — never crosses to the pipeline thread.
    #[cfg(feature = "cuda")]
    routing_stream: Option<Arc<CudaStream>>,
    /// Pinned host buffer for routing indices: `[max_tokens × k]` u32.
    /// Allocated via `cuMemAllocHost` — truly async DtoH destination.
    /// Reused every MoE layer (only one is active at a time).
    #[cfg(feature = "cuda")]
    routing_pinned: Option<PinnedRoutingBuffer>,
    /// Static GPU-native dispatch tables (all-resident cache only): per-expert
    /// weight pointers indexed on-device by `moe_bucketize`'s tile tables, so
    /// the expert forward needs no routing readback. `None` ⇒ host path.
    #[cfg(feature = "cuda")]
    gpu_dispatch: Option<GpuDispatchTables>,
    /// Set when the pipeline thread has exited (normally or by panic). Its
    /// `PipelineState` drop frees every expert slot, so the dispatch tables'
    /// captured weight pointers become dangling — the GPU-native gate checks
    /// this so a dead pipeline degrades to the host path's loud channel error
    /// instead of silently dereferencing freed VRAM. Never set in inline mode
    /// (no thread).
    pipeline_dead: Arc<AtomicBool>,
    /// Expert IDs from the most recently completed MoE layer.
    /// Used by the next layer's hint to predict via transition matrix.
    prev_layer_experts: Mutex<Vec<usize>>,
    /// Shared telemetry counters (always-on).
    stats: Arc<Mutex<PipelineStats>>,
    /// Timing accumulator for forward-thread spans (routing, submission).
    /// Only present when profiling is enabled.
    #[cfg(feature = "profile")]
    forward_profile: Mutex<ProfileAccumulator>,
}

/// Pinned host buffer for routing indices (async DtoH destination).
#[cfg(feature = "cuda")]
struct PinnedRoutingBuffer {
    /// Raw pointer from `cuMemAllocHost`.
    ptr: *mut u32,
    /// Capacity in u32 elements.
    capacity: usize,
}

#[cfg(feature = "cuda")]
impl PinnedRoutingBuffer {
    /// Allocate a pinned buffer for `capacity` u32 elements.
    fn new(capacity: usize) -> Result<Self> {
        let byte_size = capacity * std::mem::size_of::<u32>();
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = unsafe { cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, byte_size) };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            candle::bail!(
                "cuMemAllocHost for routing buffer failed: {:?} ({} bytes)",
                result,
                byte_size,
            );
        }
        Ok(Self {
            ptr: ptr as *mut u32,
            capacity,
        })
    }
}

#[cfg(feature = "cuda")]
impl Drop for PinnedRoutingBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

// SAFETY: The pinned memory is GPU-accessible and not tied to a thread.
#[cfg(feature = "cuda")]
unsafe impl Send for PinnedRoutingBuffer {}
#[cfg(feature = "cuda")]
unsafe impl Sync for PinnedRoutingBuffer {}

/// Everything [`ExpertCache::new`] needs, gathered rather than passed as nine
/// positional arguments.
///
/// `zone` is the weight side of the device reservation, already sized: its
/// capacity **is** the resident-expert count. There is no budget arithmetic left
/// at this level — `VramGovernor::expert_budget` used to divide bytes by
/// `max_expert_size` here, and the zone's capacity is that same quotient taken
/// once, against a span whose extent is a fact rather than a forecast.
pub struct ExpertCacheSetup<'a> {
    /// The GGUF, mapped. Read only while the pack is being built.
    pub mmap: Arc<memmap2::Mmap>,
    /// Per-`[layer][expert]` byte ranges into that mapping.
    pub host_refs: Vec<Vec<MmapExpertRef>>,
    /// The weight side of the device reservation.
    pub zone: WeightZone,
    pub device: &'a Device,
    pub experts_per_layer: usize,
    /// The checkpoint the experts come from — names the pack and identifies it.
    pub gguf_path: &'a std::path::Path,
    /// Where a persistent pack lives, or `None` for a temp file that is
    /// unlinked as soon as it is open and costs a repack every boot.
    pub expert_pack_dir: Option<&'a std::path::Path>,
    pub progress: Option<&'a dyn Fn(usize, usize)>,
    pub int8mode: Int8Mode,
}

impl ExpertCache {
    /// Create a new expert cache with a background pipeline thread.
    ///
    /// **CUDA path:** opens the pack file for this checkpoint — building it by
    /// repacking every expert out of the GGUF if there is not already a matching
    /// one — then fills the warm and hot tiers from it. After startup the GGUF's
    /// expert regions are never read again. Requires an actual CUDA device: a
    /// cuda-feature build handed a CPU device fails here rather than later.
    ///
    /// **Non-CUDA builds** (`not(feature = "cuda")`) take a separate path
    /// entirely, filling from the mmap with no pack and no warm tier.
    pub fn new(setup: ExpertCacheSetup<'_>) -> Result<Self> {
        let ExpertCacheSetup {
            mmap,
            host_refs,
            zone,
            device,
            experts_per_layer,
            gguf_path,
            expert_pack_dir,
            progress,
            int8mode,
        } = setup;
        // Experts run only on the KO int8 tensor-core path: the FP GEMX kernel
        // was deleted with the float fast path, so an `Off` slot would repack
        // to a layout no kernel can run and every slot construction downstream
        // would fail one expert at a time (`from_qtensor_repacked: only KO
        // twins are runnable`). Refuse here, at the one place every routed
        // model passes through, so the load fails with the reason instead of
        // the first MoE forward failing with the symptom. `Off` remains valid
        // for dense projections, which never build this cache.
        if int8mode == Int8Mode::Off {
            candle::bail!(
                "expert cache: Int8Mode::Off has no expert kernel — the FP GEMX \
                 expert path was removed, so routed (MoE) models require an int8 \
                 mode (Precision or Performance). This device/model combination \
                 selected Off; pass an explicit int8 mode that this GPU supports."
            );
        }
        let num_moe_layers = host_refs.len();
        let num_slots = zone.capacity();
        // **The pinned set must be affordable before anything is loaded.**
        //
        // `PINNED_LAYERS` is fixed, and those layers have no record in the pack
        // and no slot in the warm tier — so a zone too small to hold them plus
        // one layer's worst-case routed set does not degrade, it stops: every
        // resident slot ends up holding an expert the eviction scan is forbidden
        // to touch, and every load from then on fails permanently. The zone's
        // floor states the requirement but `WeightZone::new` does not raise a
        // smaller opening to meet it, and only one of the three MoE loaders
        // clamps its own capacity — so the check belongs here, on the path all
        // of them take.
        //
        // **CUDA only**, because a zone of zero slots is what a CPU device
        // always produces — there is no weight zone off the GPU. Checking it
        // first turned the deliberate "this build has CUDA compiled in but was
        // given a CPU device" message below into "affords 0 expert slots, below
        // the floor of 385", which names the symptom and hides the cause.
        let floor =
            minimum_resident_slots(experts_per_layer).min(num_moe_layers * experts_per_layer);
        if matches!(device, Device::Cuda(_)) && num_slots < floor {
            candle::bail!(
                "MoE expert cache: this device affords {num_slots} expert slots, below the \
                 floor of {floor} — {} permanently resident layers of {experts_per_layer} \
                 experts plus one layer's worst-case routed set. Below it the eviction scan \
                 has no candidates and every load fails.",
                pinned_layer_count(num_moe_layers),
            );
        }
        let mut inner = ExpertCacheInner::new(zone, num_moe_layers, experts_per_layer);

        // ── CUDA copy stream (pipeline thread) ──
        #[cfg(feature = "cuda")]
        let copy_stream: Option<Arc<CudaStream>> = if let Device::Cuda(cuda_dev) = device {
            match cuda_dev.cuda_context().new_stream() {
                Ok(stream) => {
                    tracing::info!("Expert cache: created copy stream for DMA overlap");
                    Some(stream)
                }
                Err(e) => {
                    tracing::warn!(
                        "Expert cache: failed to create copy stream, falling back to single-stream: {e}"
                    );
                    None
                }
            }
        } else {
            None
        };

        // ── CUDA routing stream + pinned buffer (forward thread) ──
        #[cfg(feature = "cuda")]
        let (routing_stream, routing_pinned): (
            Option<Arc<CudaStream>>,
            Option<PinnedRoutingBuffer>,
        ) = if let Device::Cuda(cuda_dev) = device {
            let stream = match cuda_dev.cuda_context().new_stream() {
                Ok(s) => {
                    tracing::info!("Expert cache: created routing stream for async DtoH");
                    Some(s)
                }
                Err(e) => {
                    tracing::warn!("Expert cache: failed to create routing stream: {e}");
                    None
                }
            };
            // 1024 tokens × 8 experts = 8192 u32 elements = 32 KB
            let buf = match PinnedRoutingBuffer::new(1024 * experts_per_layer) {
                Ok(b) => {
                    tracing::info!(
                        "Expert cache: allocated {} KB pinned routing buffer",
                        (1024 * experts_per_layer * 4) / 1024,
                    );
                    Some(b)
                }
                Err(e) => {
                    tracing::warn!("Expert cache: failed to allocate pinned routing buffer: {e}");
                    None
                }
            };
            (stream, buf)
        } else {
            (None, None)
        };

        let transition_matrix = TransitionMatrix::new(num_moe_layers, experts_per_layer);
        if num_moe_layers > 1 {
            tracing::info!(
                "Expert cache: transition matrix enabled ({} layer pairs, {} experts/layer)",
                num_moe_layers - 1,
                experts_per_layer,
            );
        }

        // ── CUDA startup: the pack, then the two resident tiers from it ──
        #[cfg(feature = "cuda")]
        let (pack, warm, cold_staging, residency, layer_geometries, all_resident) =
            if let Device::Cuda(cuda_dev) = device {
                let geoms = super::pinned::layer_geometries(&host_refs, int8mode)?;
                let total_experts = num_moe_layers * experts_per_layer;
                let all_resident = num_slots >= total_experts;

                // **The GGUF's expert regions become dead pages here.** They are
                // read once — streaming, to build the pack — and never again:
                // every later load comes from the pack, the warm pool, or VRAM.
                // The loader declared the whole mapping as resident weight
                // bytes, which is right for a dense model where the mmap *is*
                // the weight source, and wrong here by 16.6 GiB of a 18.6 GB
                // file. That reservation is subtracted from the host-RAM budget
                // the warm tier is then sized out of, so leaving it in place
                // does not merely misreport — it takes the RAM away from the
                // tier whose whole job is to stop those pages being needed.
                let expert_source_bytes: u64 = host_refs
                    .iter()
                    .flatten()
                    .map(|r| (r.gate_len + r.up_len + r.down_len) as u64)
                    .sum();
                let live_weight_bytes = (mmap.len() as u64).saturating_sub(expert_source_bytes);
                candle::vram::set_weights_mmap(live_weight_bytes);
                tracing::info!(
                    target: "candle_transformers::expert_lre",
                    mapped_gib = mmap.len() as f64 / 1e9,
                    dead_gib = expert_source_bytes as f64 / 1e9,
                    live_gib = live_weight_bytes as f64 / 1e9,
                    "expert cache: the GGUF's expert pages are the pack's job now"
                );

                // The pack's record layout **is** the VRAM slot's layout: same
                // three projections, same aligned offsets. One geometry, so a
                // load is a read and a copy with nothing rearranged in between.
                let slot_bytes = slot_bytes_for(&geoms);
                let layers: Vec<LayerSpansInput> = geoms
                    .iter()
                    .map(|g| {
                        let (gate, up, down, _) = slot_offsets(g);
                        LayerSpansInput {
                            gate: (gate, g.gate_repacked_size, g.gate_dtype),
                            up: (up, g.up_repacked_size, g.up_dtype),
                            down: (down, g.down_repacked_size, g.down_dtype),
                        }
                    })
                    .collect();
                let layouts: Vec<RecordLayout> =
                    layers.iter().copied().map(RecordLayout::from).collect();
                // Run the repack over a reference matrix in every quantisation
                // the engine supports, and hash it. The pack's validity is then
                // checked against what this build *produces* and not only
                // against where it would put it — see `pack::fingerprint`.
                let source = open_or_create(PackSpec {
                    dir: expert_pack_dir,
                    gguf_path,
                    identity: PackIdentity::of(&mmap, int8mode, repack_fingerprint(cuda_dev)),
                    num_layers: num_moe_layers,
                    experts_per_layer,
                    // The leading layers the cache pins permanently. They are
                    // never evicted, so they are never reloaded, so the pack
                    // holds no records for them.
                    pinned_layers: pinned_layer_count(num_moe_layers),
                    slot_bytes,
                    layers,
                })?;

                // The warm tier is sized by what the machine can spare, not by
                // what residency demands — the cold tier serves every expert at
                // any warm size, including zero.
                let stride = candle::direct_io::round_up_sector(slot_bytes);
                // **Before the warm tier, not after.** This is 46 MB and
                // mandatory; the warm tier is ~14 GB and elastic. Taking the
                // elastic one first left this to fail on a machine the warm
                // tier had just filled — a model load dying with
                // `CUDA_ERROR_OUT_OF_MEMORY` where it should have been a
                // slightly smaller warm tier. The order is the fix; the
                // aligned-host fallback inside is the belt.
                let cold_staging = ColdStaging::new(stride, COLD_STAGING_BUFFERS)?;
                // **A cache that holds every expert in VRAM wants no warm tier
                // at all.** Nothing is ever evicted in that state — `post_compute`
                // returns before the eviction and boundary passes — so a warm
                // slot could only ever be read if a load missed, and no load
                // does. Sizing it anyway would pin the model's size in host RAM
                // to serve nothing, and pay a full-pack read at startup for it.
                // The warm tier's job is covering **misses**, and the pinned
                // prefix never generates one, so it is sized against the
                // evictable set rather than the model. On the 3.6-35B that is
                // 512 fewer experts to aim at — 943 MiB of pinned host memory
                // that used to be spent on experts no load could ever ask for.
                let pinned = pinned_layer_count(num_moe_layers);
                let evictable = total_experts - pinned * experts_per_layer;
                let want_warm = if all_resident {
                    0
                } else {
                    warm_slots_for(stride, evictable)
                };
                // `num_slots` is exactly what the startup fill will take into
                // VRAM, in flat order, so it is the prefix the draw skips over.
                let membership = stratified_membership(
                    num_moe_layers,
                    experts_per_layer,
                    want_warm,
                    num_slots,
                    pinned,
                    WARM_DRAW_SEED,
                );
                let mut warm = WarmPool::new(membership.len(), stride);
                // A refusal shortens the draw rather than leaving slots the pool
                // does not have: `ram` must never name a slot outside it.
                let membership = &membership[..membership.len().min(warm.num_slots())];

                let mut residency =
                    vec![vec![ExpertResidency::default(); experts_per_layer]; num_moe_layers];
                // The eviction policy weighs what a reload would cost, so it has
                // to know which experts the warm tier holds before the first
                // victim is chosen.
                inner.set_warm_backed(membership);
                let targets = StartupTargets {
                    inner: &mut inner,
                    warm: &mut warm,
                    residency: &mut residency,
                    membership,
                    geoms: &geoms,
                    layouts: &layouts,
                    stride,
                    mmap: &mmap,
                    host_refs: &host_refs,
                };
                let pack = match source {
                    PackSource::Ready(pack) => {
                        startup_from_pack(
                            targets,
                            &pack,
                            num_moe_layers,
                            experts_per_layer,
                            cuda_dev,
                            progress,
                        )?;
                        pack
                    }
                    PackSource::Build(mut writer) => {
                        startup_repack(targets, &mut writer, cuda_dev, progress)?;
                        writer.finish()?
                    }
                };

                // The pack's stride is what the geometry said it would be — the
                // buffers above were cut to it before the file was opened.
                debug_assert_eq!(pack.stride(), stride);
                (pack, warm, cold_staging, residency, geoms, all_resident)
            } else {
                // **A cuda-feature build has no expert path for a CPU device,
                // and never had one.** Every tier is device-side or DMA-bound:
                // the hot tier is weight-zone slots, the warm tier is pinned
                // host memory the GPU reads, the pack's records are uploaded
                // with `cuMemcpyHtoD`. The `not(feature = "cuda")` build has a
                // separate mmap path (`prewarm_expert_cache`); this build does
                // not, and cannot borrow it — the zone it would fill has no
                // device reservation behind it.
                //
                // This used to construct an empty pool and empty tables and
                // return successfully, which meant the load *appeared* to work
                // and the first MoE layer panicked indexing an empty
                // per-layer table. Failing here says the same thing at the
                // point where it can still be acted on.
                candle::bail!(
                    "MoE expert cache: this build has CUDA compiled in but was given {device:?}. \
                     The expert tiers are all device-side or DMA-bound, so there is no CPU path — \
                     run on a CUDA device, or build without the `cuda` feature for the mmap path."
                )
            };

        // ── Non-CUDA prewarm path ──
        #[cfg(not(feature = "cuda"))]
        {
            let _ = progress; // not yet wired into the legacy prewarm path
            prewarm_expert_cache(&mut inner, &mmap, &host_refs, device, int8mode);
        }

        // Non-CUDA: all_resident is always false (no VRAM cache to fill).
        #[cfg(not(feature = "cuda"))]
        let all_resident = false;

        let stats = PipelineStats::new_shared();
        // Seed the resident-expert VRAM gauge with the startup footprint (occupied
        // slots × slot size) so it reads correctly before the first classify
        // refreshes it. `inner` + `layer_geometries` are still in scope here,
        // before they move into `PipelineState` below.
        #[cfg(feature = "cuda")]
        {
            let occupied = inner.num_slots() - inner.free_len();
            let slot_bytes = layer_geometries
                .iter()
                .map(|g| g.total_repacked_size)
                .max()
                .unwrap_or(0);
            let seeded = occupied * slot_bytes;
            tracing::info!(
                target: "candle_transformers::expert_lre",
                num_slots = inner.num_slots(),
                free_slots = inner.free_len(),
                occupied,
                slot_bytes,
                resident_gib = seeded as f64 / 1e9,
                "expert cache: seeded resident-VRAM gauge"
            );
            if let Ok(mut s) = stats.lock() {
                s.resident_vram_bytes = seeded;
                s.warm_slots = warm.num_slots();
                s.total_experts = num_moe_layers * experts_per_layer;
            }
        }

        // With every expert staged into a permanent VRAM slot by the
        // (synchronous) startup above, the weight addresses are static:
        // capture them into device pointer tables while `inner` is still in
        // scope, so the expert forward dispatches entirely on-GPU (no per-layer
        // routing readback, no pipeline-thread handoff). Only meaningful when
        // all-resident — a paged cache's pointers move with eviction.
        #[cfg(feature = "cuda")]
        let gpu_dispatch = if all_resident {
            if let Device::Cuda(cuda_dev) = device {
                GpuDispatchTables::build(&inner, cuda_dev)
            } else {
                None
            }
        } else {
            None
        };

        // Arc-share the two source tiers and the geometry table with the
        // off-thread expert streamer (both are immutable from here on), and
        // spawn it — it owns its own staging ring and CUDA stream, so a
        // whole-layer prefill stream never runs its reads on the pipeline
        // thread. All-resident caches have nothing to stream.
        #[cfg(feature = "cuda")]
        let pack = Arc::new(pack);
        #[cfg(feature = "cuda")]
        let warm = Arc::new(warm);
        #[cfg(feature = "cuda")]
        let layer_geometries = Arc::new(layer_geometries);
        #[cfg(feature = "cuda")]
        let streamer = if all_resident {
            None
        } else if let Device::Cuda(cuda_dev) = device {
            match cuda_dev.cuda_context().new_stream() {
                Ok(stream) => {
                    super::streamer::spawn_streamer_thread(super::streamer::StreamerCtx {
                        pack: pack.clone(),
                        warm: warm.clone(),
                        layer_geometries: layer_geometries.clone(),
                        cuda_dev: cuda_dev.clone(),
                        stream,
                        stats: stats.clone(),
                    })
                }
                Err(e) => {
                    tracing::warn!(
                        target: "candle_transformers::expert_lre",
                        "streamer stream unavailable ({e}); expert streaming disabled"
                    );
                    None
                }
            }
        } else {
            None
        };

        let state = PipelineState {
            inner,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            copy_stream,
            #[cfg(feature = "cuda")]
            pack,
            #[cfg(feature = "cuda")]
            warm,
            #[cfg(feature = "cuda")]
            cold_staging,
            #[cfg(feature = "cuda")]
            residency,
            #[cfg(feature = "cuda")]
            layer_geometries,
            num_moe_layers,
            #[cfg(feature = "cuda")]
            all_resident,
            transition_matrix,
            last_moe_layer_idx: None,
            speculative_loads: HashSet::new(),
            #[cfg(feature = "cuda")]
            streamer,
            #[cfg(feature = "cuda")]
            pending_streams: HashMap::new(),
            #[cfg(feature = "cuda")]
            stream_loads: HashSet::new(),
            prefetch_fences: Vec::new(),
            prefetch_depth: 1,
            pass_started: None,
            pass_dma_bytes: 0,
            pass_late: 0,
            pass_pred: (0, 0),
            bw_ceiling_gbps: 0.0,
            hint_stats: (0, 0),
            profile: ProfileAccumulator::new(),
            stats: stats.clone(),
            #[cfg(not(feature = "cuda"))]
            mmap,
            #[cfg(not(feature = "cuda"))]
            host_refs,
            #[cfg(not(feature = "cuda"))]
            int8mode,
        };

        // Cloned before `state` moves to the pipeline thread — this handle is
        // the reader that could not otherwise see the zone at all.
        let geometry = state.inner.geometry.clone();
        let pipeline_dead = Arc::new(AtomicBool::new(false));
        let tx = spawn_pipeline_thread(state, pipeline_dead.clone());

        Ok(Self {
            mode: PipelineMode::Threaded { tx },
            all_resident,
            geometry,
            #[cfg(feature = "cuda")]
            routing_stream,
            #[cfg(feature = "cuda")]
            routing_pinned,
            #[cfg(feature = "cuda")]
            gpu_dispatch,
            pipeline_dead,
            prev_layer_experts: Mutex::new(Vec::new()),
            stats,
            #[cfg(feature = "profile")]
            forward_profile: Mutex::new(ProfileAccumulator::new()),
        })
    }

    /// Construct an `ExpertCache` in inline mode with pre-populated slots.
    ///
    /// Used when all experts are already loaded to VRAM (e.g. the non-mmap
    /// reader path).  No background thread, no copy stream, no DMA.
    pub fn new_prepopulated(
        slots: Vec<Option<ExpertSlot>>,
        key_to_slot: HashMap<(usize, usize), usize>,
        last_used: Vec<u32>,
        generation: u32,
        slot_to_key: Vec<Option<(usize, usize)>>,
        device: &Device,
    ) -> Self {
        // Every slot is occupied and none ever moves, so the zone exists only to
        // answer "which index is where" — the addresses are the pre-built
        // storages' own, not the zone's, and nothing here allocates from it. The
        // floor is the whole capacity for the same reason: nothing may retract a
        // zone whose slots it does not own.
        let mut zone = WeightZone::new(0, 0, slots.len(), slots.len(), slots.len());
        for _ in 0..slots.len() {
            zone.alloc();
        }
        let slot_count = zone.capacity();
        let zone_frontier = zone.slot_base(slot_count.saturating_sub(1));
        let inner = ExpertCacheInner {
            slots,
            zone,
            key_to_slot,
            last_used,
            generation,
            // Nothing may retract a zone whose slot addresses it does not own,
            // so this geometry is fixed for the object's life — published all
            // the same, so the staleness check reads one shape of thing in both
            // modes rather than being conditional on which constructor ran.
            geometry: Arc::new(ZoneGeometry::new(
                slot_count,
                zone_frontier,
            )),
            slot_to_key,
            expert_scores: vec![],
            num_moe_layers: 0,
            experts_per_layer: 0,
            // Nothing is ever evicted here, so nothing needs protecting from
            // eviction either.
            pinned_layers: 0,
            // Inline mode holds every expert in VRAM and never evicts, so no
            // reload cost is ever weighed.
            warm_backed: vec![],
        };
        // All experts are VRAM-resident and never move, so their weight
        // addresses are static: capture them once into device pointer tables
        // and the expert forward dispatches entirely on-GPU (no per-layer
        // routing readback). Best-effort — `None` keeps the host path.
        #[cfg(feature = "cuda")]
        let gpu_dispatch = if let Device::Cuda(cuda_dev) = device {
            GpuDispatchTables::build(&inner, cuda_dev)
        } else {
            None
        };
        let geometry = inner.geometry.clone();
        Self {
            mode: PipelineMode::Inline {
                inner: Mutex::new(inner),
                device: device.clone(),
            },
            all_resident: true, // prepopulated = all in VRAM
            geometry,
            #[cfg(feature = "cuda")]
            routing_stream: None,
            #[cfg(feature = "cuda")]
            routing_pinned: None,
            #[cfg(feature = "cuda")]
            gpu_dispatch,
            pipeline_dead: Arc::new(AtomicBool::new(false)),
            prev_layer_experts: Mutex::new(Vec::new()),
            stats: PipelineStats::new_shared(),
            #[cfg(feature = "profile")]
            forward_profile: Mutex::new(ProfileAccumulator::new()),
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Public API
    // ────────────────────────────────────────────────────────────────────────

    /// Live VRAM bytes held by resident expert slots (`occupied_slots ×
    /// slot_size`) — the model's **time-varying** MoE weight footprint. Rises as
    /// experts load into VRAM and falls as they stream out to pinned RAM under
    /// pressure. Read lock-free from the shared stats gauge (seeded at
    /// construction, refreshed each classify). `0` on non-CUDA / no-expert models.
    pub fn resident_vram_bytes(&self) -> usize {
        PipelineStats::snapshot(&self.stats).resident_vram_bytes
    }

    /// Submit a full MoE dispatch to the pipeline.
    ///
    /// This is the primary entry point for `SparseMoeBlock::forward()`.
    /// In threaded mode, sends the work to the background thread and blocks
    /// until the result arrives.  In inline mode, executes synchronously.
    ///
    /// # Arguments
    ///
    /// * `moe_layer_idx` — which MoE layer (0..num_moe_layers)
    /// * `expert_ids` — unique expert IDs selected by the router (sorted)
    /// * `xs` — input hidden states `[num_tokens, hidden_dim]`
    /// * `weights_flat` — flattened routing weights `[num_tokens * k]`
    /// * `assignments` — flat sorted `(expert_id, token_idx, weight_idx)` array
    ///
    /// # Returns
    ///
    /// Output tensor `[num_tokens, hidden_dim]` — the weighted sum of expert outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_moe_work(
        &self,
        moe_layer_idx: usize,
        expert_ids: Vec<usize>,
        input: MoeInput,
        out_dtype: DType,
        weights_flat: &Tensor,
        assignments: Vec<(u32, u32, u32)>,
        wave: Option<WaveTicket>,
    ) -> Result<Tensor> {
        match &self.mode {
            PipelineMode::Threaded { tx } => {
                let t = profile_now();

                // Create a oneshot response channel.
                let (resp_tx, resp_rx) = mpsc::sync_channel(1);

                let request = MoeWorkRequest {
                    moe_layer_idx,
                    expert_ids,
                    input,
                    out_dtype,
                    weights_flat: weights_flat.clone(),
                    assignments,
                    // Captured right before `send` so the worker can split the inbound handoff
                    // (channel wakeup) out of the actual work.
                    wave,
                    submitted_at: profile_now(),
                    response_tx: resp_tx,
                };

                tx.send(PipelineMessage::Work(request)).map_err(|_| {
                    candle::Error::Msg("Expert pipeline thread died — channel closed".into())
                })?;

                // Block until the pipeline thread returns the result + its completion timestamp.
                let (result, worker_done_at) = resp_rx.recv().map_err(|_| {
                    candle::Error::Msg(
                        "Expert pipeline thread died — response channel closed".into(),
                    )
                })?;

                // submit_roundtrip = submit_inbound (worker) + pipe_worker_total (worker) +
                // submit_outbound (here). The two handoffs are the reclaimable threading tax;
                // pipe_worker_total is the irreducible classify + DMA + compute.
                self.record_profile("submit_outbound", worker_done_at);
                self.record_profile("submit_roundtrip", t);
                result
            }
            PipelineMode::Inline { inner, device } => {
                // Inline mode: no thread, just lock and compute.
                self.submit_inline(
                    inner,
                    device,
                    moe_layer_idx,
                    &expert_ids,
                    input,
                    out_dtype,
                    weights_flat,
                    &assignments,
                    wave,
                )
            }
        }
    }

    /// Inline execution path (reader/prepopulated mode).
    ///
    /// All experts are already resident — no DMA, no copy stream.
    /// Lock the Mutex (uncontended), compute all experts by slot index.
    #[allow(clippy::too_many_arguments)]
    fn submit_inline(
        &self,
        mutex: &Mutex<ExpertCacheInner>,
        device: &Device,
        moe_layer_idx: usize,
        expert_ids: &[usize],
        input: MoeInput,
        out_dtype: DType,
        weights_flat: &Tensor,
        assignments: &[(u32, u32, u32)],
        wave: Option<WaveTicket>,
    ) -> Result<Tensor> {
        let mut inner = mutex
            .lock()
            .map_err(|_| candle::Error::Msg("ExpertCache Mutex poisoned".into()))?;

        // In inline mode, all experts are resident — classify as all hits.
        let mut hits: Vec<(usize, usize)> = Vec::with_capacity(expert_ids.len());
        for &expert_idx in expert_ids {
            if let Some(&slot_idx) = inner.key_to_slot.get(&(moe_layer_idx, expert_idx)) {
                inner.promote(slot_idx);
                hits.push((expert_idx, slot_idx));
            } else {
                return Err(candle::Error::Msg(format!(
                    "Inline mode: expert ({moe_layer_idx}, {expert_idx}) not found in cache"
                )));
            }
        }

        let expert_group = |eid: usize| -> (Vec<u32>, Vec<u32>) {
            let eid32 = eid as u32;
            let lo = assignments.partition_point(|a| a.0 < eid32);
            let hi = assignments.partition_point(|a| a.0 <= eid32);
            let toks: Vec<u32> = assignments[lo..hi].iter().map(|a| a.1).collect();
            let wids: Vec<u32> = assignments[lo..hi].iter().map(|a| a.2).collect();
            (toks, wids)
        };

        let (num_tokens, hidden) = input.shape()?;
        // The inline twin of the threaded pipeline's combine target. It runs on
        // the caller's thread and stream, so the layer's FFN generation *would*
        // bound it — but this mode exists only when there is no pipeline thread,
        // which production never configures, so it stays an ordinary allocation
        // rather than a second wave consumer for a path nothing takes.
        let mut ys = Tensor::zeros((num_tokens, hidden), out_dtype, device)?;
        let mut _inline_prof = ProfileAccumulator::new();
        #[cfg(feature = "cuda")]
        {
            let experts_data: Vec<(Vec<u32>, Vec<u32>)> =
                hits.iter().map(|&(eidx, _)| expert_group(eidx)).collect();
            // A missing slot is a DROPPED expert, not a skippable one: the old
            // `filter_map(… as_ref()?)` silently omitted it, so its contribution
            // vanished and the layer returned an answer computed from fewer than
            // k experts, indistinguishable downstream from a correct one. Same
            // hazard, same refusal, as the threaded pipeline's copy.
            let mut experts_vec: Vec<(&ExpertSlot, &[u32], &[u32])> =
                Vec::with_capacity(hits.len());
            for (&(eidx, slot_idx), (toks, wids)) in hits.iter().zip(experts_data.iter()) {
                let Some(slot) = inner.slots[slot_idx].as_ref() else {
                    candle::bail!(
                        "expert {eidx} was classified resident in slot {slot_idx}, but that slot \
                         is empty at compute time: its contribution would be dropped from this \
                         layer's output"
                    );
                };
                experts_vec.push((slot, toks.as_slice(), wids.as_slice()));
            }
            if !experts_vec.is_empty() {
                compute_experts_grouped(
                    &input,
                    &mut ys,
                    &experts_vec,
                    weights_flat,
                    &mut _inline_prof,
                    wave,
                )?;
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Non-CUDA only ever sees `Float` (int8/q8a128 is cuda-only).
            let MoeInput::Float(xs) = &input;
            for &(eidx, slot_idx) in &hits {
                let slot = inner.slots[slot_idx].as_ref().ok_or_else(|| {
                    candle::Error::Msg(format!("inline slot {slot_idx} unexpectedly empty"))
                })?;
                let (toks, w_ids) = expert_group(eidx);
                compute_expert_contribution_gpu_weights(
                    xs,
                    &mut ys,
                    slot,
                    &toks,
                    weights_flat,
                    &w_ids,
                )?;
            }
        }

        Ok(ys)
    }

    /// Legacy API: classify and load experts, returning slot indices + fence.
    ///
    /// Only available in inline mode.  Panics in threaded mode (use
    /// `submit_moe_work` instead).
    pub fn classify_and_load(
        &self,
        moe_idx: usize,
        expert_ids: &[usize],
    ) -> Result<ClassifiedExperts> {
        match &self.mode {
            PipelineMode::Inline { inner, .. } => {
                let mut guard = inner
                    .lock()
                    .map_err(|_| candle::Error::Msg("ExpertCache Mutex poisoned".into()))?;

                let mut hits: Vec<(usize, usize)> = Vec::new();
                for &expert_idx in expert_ids {
                    if let Some(&slot_idx) = guard.key_to_slot.get(&(moe_idx, expert_idx)) {
                        if guard.slots[slot_idx].is_some() {
                            guard.promote(slot_idx);
                            hits.push((expert_idx, slot_idx));
                        }
                    }
                }

                Ok(ClassifiedExperts {
                    hits,
                    loaded: vec![],
                    fence: CopyBatchFence::noop(),
                })
            }
            PipelineMode::Threaded { .. } => Err(candle::Error::Msg(
                "classify_and_load not available in threaded mode — use submit_moe_work".into(),
            )),
        }
    }

    /// Lock the inner cache state and run a closure with mutable access.
    ///
    /// Only available in inline mode.  Panics in threaded mode.
    pub fn with_inner<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut ExpertCacheInner) -> Result<R>,
    {
        match &self.mode {
            PipelineMode::Inline { inner, .. } => {
                let mut guard = inner
                    .lock()
                    .map_err(|_| candle::Error::Msg("ExpertCache Mutex poisoned".into()))?;
                f(&mut guard)
            }
            PipelineMode::Threaded { .. } => Err(candle::Error::Msg(
                "with_inner not available in threaded mode — use submit_moe_work".into(),
            )),
        }
    }

    /// Make the main (compute) stream wait for a batch fence.
    ///
    /// Only needed when using the legacy classify_and_load API.
    /// In threaded mode, fence waiting is handled by the pipeline thread.
    pub fn wait_for_copies(&self, fence: CopyBatchFence) -> Result<()> {
        let _fence = fence;
        #[cfg(feature = "cuda")]
        if let Some(ref event) = _fence.event {
            let device = match &self.mode {
                PipelineMode::Inline { device, .. } => device,
                PipelineMode::Threaded { .. } => {
                    return Err(candle::Error::Msg(
                        "wait_for_copies not available in threaded mode".into(),
                    ));
                }
            };
            if let Device::Cuda(cuda_dev) = device {
                cuda_dev
                    .cuda_stream()
                    .wait(event)
                    .map_err(candle::Error::wrap)?;
            }
        }
        Ok(())
    }

    /// Returns true if this cache uses the background pipeline thread.
    pub fn is_threaded(&self) -> bool {
        matches!(self.mode, PipelineMode::Threaded { .. })
    }

    /// Send a speculative prediction hint to the pipeline thread.
    ///
    /// The hint tells the pipeline thread which MoE layer is coming and
    /// which experts were used in the previous layer.  The pipeline thread
    /// uses the transition matrix to predict which experts will be needed
    /// and starts DMA before the full work request arrives.
    ///
    /// No-op in inline mode, when all experts are resident, or if the
    /// channel send fails.
    pub fn send_hint(&self, layer_idx: usize, prev_expert_ids: Vec<usize>) {
        if self.all_resident {
            return;
        }
        if let PipelineMode::Threaded { tx } = &self.mode {
            let _ = tx.send(PipelineMessage::Hint {
                layer_idx,
                prev_expert_ids,
            });
        }
    }

    /// Get the routing stream for async DtoH (if available).
    #[cfg(feature = "cuda")]
    pub fn routing_stream(&self) -> Option<&Arc<CudaStream>> {
        self.routing_stream.as_ref()
    }

    /// The static GPU-native dispatch tables, when this cache is all-resident
    /// and they were successfully built at construction. `Some` ⇒ the expert
    /// forward can run entirely on-device (no routing readback).
    #[cfg(feature = "cuda")]
    pub fn gpu_dispatch(&self) -> Option<&GpuDispatchTables> {
        self.gpu_dispatch.as_ref()
    }

    /// Whether the pipeline thread has exited — after which the dispatch
    /// tables' captured weight pointers are dangling and the GPU-native path
    /// must not run. Always `false` in inline mode.
    pub fn pipeline_dead(&self) -> bool {
        self.pipeline_dead.load(Ordering::Acquire)
    }

    /// The dispatch tables IF the GPU-native path is safe for `moe_layer_idx`
    /// with a router `n_experts` wide — the cache-owned safety conditions in
    /// one place so no caller can skip part of the chain:
    /// * tables built (all-resident, KO weights, null-stream compute),
    /// * the layer inside the covered row range,
    /// * router width == table row width (a mismatch would index other
    ///   layers' experts),
    /// * the pipeline thread alive (its death frees the slot weights the
    ///   tables point at — dangling-pointer dispatch).
    ///
    /// Callers still gate their own model-level conditions (top-k bound,
    /// routing capture, diagnostic env overrides).
    #[cfg(feature = "cuda")]
    pub fn live_gpu_dispatch(
        &self,
        moe_layer_idx: usize,
        n_experts: usize,
    ) -> Option<&GpuDispatchTables> {
        // Each refusal names itself, ONCE. `GpuDispatchTables::build` already
        // reports why the tables could not be made; this is the other half —
        // tables that exist but are not usable for this call — and it was the
        // silent half. Costing decode a multiple because a per-layer condition
        // quietly disagreed is not something to learn from a profiler three
        // crates away.
        let Some(gd) = self.gpu_dispatch.as_ref() else {
            // Deliberately does NOT guess why. A paged cache never builds
            // tables by design, and an all-resident one that failed to has
            // already said so with the actual reason — asserting one of those
            // here would put a confident wrong answer next to the right one.
            log_dispatch_refusal("no dispatch tables were built for this cache", String::new);
            return None;
        };
        if gd.expert_base(moe_layer_idx).is_none() {
            log_dispatch_refusal("MoE layer is outside the covered table range", || {
                format!("layer {moe_layer_idx}")
            });
            return None;
        }
        if gd.n_experts != n_experts {
            log_dispatch_refusal("router width disagrees with the tables", || {
                format!("router {n_experts} vs tables {}", gd.n_experts)
            });
            return None;
        }
        if self.pipeline_dead() {
            log_dispatch_refusal("expert pipeline thread is dead", String::new);
            return None;
        }
        // **The tables' addresses are only valid while the zone still owns
        // them.**
        //
        // They are captured once, on the reasoning that an all-resident cache's
        // weight addresses are static. All-resident at startup is not resident
        // for ever: `WeightZone::retract_to` concedes slots to the KV side under
        // pressure, the KV arena allocates that ground and writes to it, and a
        // table entry pointing there stops naming an expert weight. The GEMM
        // then reads KV bytes as weights — finite, plausibly-shaped, and wrong,
        // which is why it surfaces as a NaN several layers downstream rather
        // than as a fault.
        //
        // The host path has always refused a weight pointer below the live
        // `weight_floor` ("that is KV ground, not an expert slot"). This is the
        // GPU-native half of the same rule, and its absence is what let a
        // retraction corrupt a contiguous run of expert slots silently.
        // Read through the shared `ZoneGeometry`, not through `inner`.
        //
        // This check was originally written as
        // `if let PipelineMode::Inline { inner, .. } = &self.mode`, justified by
        // "an all-resident cache is `Inline` by construction". It is not:
        // `ExpertCache::new` builds the dispatch tables when `all_resident` and
        // then returns `PipelineMode::Threaded`, having moved `inner` into the
        // pipeline thread's state. The branch never matched, so the guard was
        // dead code in exactly the configuration it was written for, and a
        // retraction went undetected. Measured: an expert weight the tables
        // still named sitting 31 MiB BELOW the live `weight_floor` — KV ground
        // the arena had already allocated and written — read back by the GEMM as
        // an expert weight and surfacing as NaN eleven layers later.
        //
        // The geometry is published lock-free by whichever thread owns the zone,
        // so this is mode-independent and costs three relaxed loads per layer.
        //
        // **The refusal is permanent, and that is the correct behaviour — do not
        // "fix" it by re-trusting the tables once the zone grows back.** The
        // tempting reading is that a concede-then-regrow restores the geometry,
        // so the cached addresses become valid again. The addresses do; the
        // MAPPING does not. `ExpertCacheInner::evict` drops the key → slot
        // entry, and re-admission takes the rightmost free slot
        // (`allocate_slot`), so an expert comes back at a DIFFERENT address
        // while the tables still name its old one. Re-taking them is the
        // corruption this guard was written for, wearing a recovery's clothes.
        //
        // Rebuilding the tables against the live mapping would be sound at the
        // instant of the rebuild, and is the only real recovery — but only
        // while the cache is all-resident again, because in a paged cache every
        // admission moves an expert and the tables would need invalidating per
        // admission. So the cost of a concession is the host path for the rest
        // of the process (measured elsewhere at roughly 18-19 ms → 26-28 ms per
        // decode step). That is a real price, and it is the price of being
        // right: before this check was reachable, the "faster" path was reading
        // KV bytes as expert weights.
        if let Some((now_capacity, now_floor)) = gd.zone_moved(&self.geometry) {
            log_dispatch_refusal(
                "the weight zone has retracted since the tables were built — their slot \
                 addresses now name KV ground",
                || {
                    format!(
                        "capacity {} → {now_capacity}, floor {:#x} → {now_floor:#x}",
                        gd.built_capacity(),
                        gd.built_floor(),
                    )
                },
            );
            return None;
        }
        Some(gd)
    }

    /// Base pointer of the pinned routing buffer, if it exists and holds `len`.
    ///
    /// A pointer rather than a `&mut [u32]`: the cache lives behind an `Arc`, so
    /// a `&mut` handed out from `&self` is one the borrow checker cannot police —
    /// nothing stops a second call from minting an overlapping one. The single
    /// writer here is the forward thread, and the ordering against the routing
    /// stream's DtoH is the caller's (it holds the events), so the caller is
    /// where the slice and its `unsafe` belong.
    #[cfg(feature = "cuda")]
    pub fn routing_pinned_ptr(&self, len: usize) -> Option<*mut u32> {
        let pinned = self.routing_pinned.as_ref()?;
        if len > pinned.capacity {
            return None;
        }
        Some(pinned.ptr)
    }

    /// Store the expert IDs from the most recently completed MoE layer.
    /// Called after routing indices are read and before submitting work.
    pub fn set_prev_layer_experts(&self, experts: Vec<usize>) {
        if let Ok(mut prev) = self.prev_layer_experts.lock() {
            *prev = experts;
        }
    }

    /// Get a clone of the previous layer's expert IDs for hint prediction.
    pub fn get_prev_layer_experts(&self) -> Vec<usize> {
        self.prev_layer_experts
            .lock()
            .map_or_else(|_| vec![], |v| v.clone())
    }

    /// Buy `regions` of weight-side ground for the KV side, and answer with the
    /// bytes it conceded.
    ///
    /// **The caller states the quantity.** It is either an arena claim that has
    /// run the KV side out and is asking for what it is about to allocate, or the
    /// scheduler's relief asking for its measured setpoint shortfall. Both know
    /// the number; neither can accumulate one, because the request does not
    /// outlive the call that made it. What this replaced — a running count of
    /// refused claims drained here — could and did: 4,436 regions against a
    /// twenty-eight-region need, paid in full.
    ///
    /// **For a caller that is stuck.** The other direction — the weight side
    /// taking back ground the KV side is not using — only runs between forwards
    /// ([`Self::reclaim_spare_ground`]), and a KV side that cannot allocate the
    /// arenas a wave needs never reaches the next one. This is the path that
    /// breaks that.
    ///
    /// Zero is an ordinary answer: the zone may already sit at its floor, or a
    /// wave generation may still be open, in which case `set_weight_floor`
    /// refuses and this reports that it did. The caller decides whether to retry
    /// on that basis rather than spinning on a claim that cannot succeed.
    ///
    /// Blocks on the pipeline thread's reply — it owns the cache state and is
    /// the only place a boundary move is safe.
    pub fn request_kv_ground(&self, regions: usize) -> u64 {
        let PipelineMode::Threaded { tx } = &self.mode else {
            // Inline mode holds every expert in VRAM and never moves the
            // boundary; there is no ground to offer.
            return 0;
        };
        if regions == 0 {
            return 0;
        }
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        if tx
            .send(PipelineMessage::RenegotiateBoundary {
                regions,
                response_tx,
            })
            .is_err()
        {
            return 0;
        }
        response_rx.recv().unwrap_or(0)
    }

    /// The other direction: take back KV regions that are standing free.
    ///
    /// **Call this only between forwards.** Moving the boundary evicts and
    /// relocates expert slots, and a wave in flight may be reading either, so
    /// `set_weight_floor` refuses while a wave generation is open on the span.
    ///
    /// This used to be driven from the pipeline thread's `post_compute`, which
    /// runs the instant a MoE layer's work is answered — with the forward thread
    /// still inside `ffn_forward` holding that layer's FFN wave guard. So it was
    /// asked forty-eight times a forward from inside the wave, and whether it
    /// landed came down to a race with the forward thread's phase transitions:
    /// refused in the common case, and in the narrow window between one layer's
    /// guard dropping and the next one's opening, granted — at the cost of a
    /// device-wide quiesce in the middle of a forward. Neither outcome is one the
    /// engine should depend on, which is why the caller is now the wave loop's
    /// own inter-forward gap, alongside the transient tier's hand-back.
    ///
    /// Answers with the bytes taken — always zero, since this direction concedes
    /// nothing; the value exists so the two directions share a signature.
    pub fn reclaim_spare_ground(&self) -> u64 {
        let PipelineMode::Threaded { tx } = &self.mode else {
            return 0;
        };
        // **Sweep before asking, because the answer is computed from `live`.**
        //
        // `spare_regions` reads `live + tier` as the KV side's demand and feeds
        // it to a sixty-second window peak. A region whose arena went chunk-empty
        // several waves ago is still `live` until something sweeps it, so without
        // this the negotiation is answered from a demand figure that includes
        // arenas holding nothing — and the inflated peak then suppresses growth
        // for a full window after the ground actually came free.
        //
        // The reactive sweeps cannot cover it: `claim_region` sweeps only when
        // the free list is empty and `place_transient` only after a placement
        // came up short, and a workload with spare regions reaches neither. The
        // 3.6-35B gate runs at `free 18` throughout, so nothing swept between
        // configs at all.
        //
        // Here rather than in each model's wave loop so every MoE model gets it,
        // and so the sweep and the question it informs cannot drift apart.
        #[cfg(feature = "cuda")]
        candle_nn::kv_cache::reclaim_empty_arenas();
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        // Zero regions is the growth question — "how much is the KV side holding
        // that I could take?" — as against a positive count, which is the KV side
        // stating what it needs.
        if tx
            .send(PipelineMessage::RenegotiateBoundary {
                regions: 0,
                response_tx,
            })
            .is_err()
        {
            return 0;
        }
        response_rx.recv().unwrap_or(0)
    }

    /// Snapshot and reset all profile accumulators (forward + pipeline threads).
    ///
    /// Returns a merged [`ProfileSnapshot`] containing both forward-thread
    /// spans (prefixed `fwd_`) and pipeline-thread spans (`pipe_`, `cl_`,
    /// `gemm_`, `dma_`).
    pub fn snapshot_profiles(&self) -> ProfileSnapshot {
        // Forward thread profile (lives on this handle behind Mutex).
        let fwd = {
            #[cfg(feature = "profile")]
            {
                self.forward_profile.lock().map_or_else(
                    |_| ProfileSnapshot::default(),
                    |mut prof| {
                        let snap = prof.snapshot();
                        prof.reset();
                        snap
                    },
                )
            }
            #[cfg(not(feature = "profile"))]
            {
                ProfileSnapshot::default()
            }
        };

        // Pipeline thread profile (request via message).
        let pipe = match &self.mode {
            PipelineMode::Threaded { tx } => {
                let (resp_tx, resp_rx) = std::sync::mpsc::sync_channel(1);
                if tx
                    .send(PipelineMessage::SnapshotProfile {
                        response_tx: resp_tx,
                    })
                    .is_ok()
                {
                    resp_rx.recv().unwrap_or_default()
                } else {
                    ProfileSnapshot::default()
                }
            }
            PipelineMode::Inline { .. } => ProfileSnapshot::default(),
        };

        // Merge both into a single snapshot.
        let mut merged = fwd;
        merged.merge(&pipe);
        merged
    }

    /// Snapshot the current pipeline telemetry counters.
    pub fn expert_stats(&self) -> PipelineStats {
        PipelineStats::snapshot(&self.stats)
    }

    /// Span bytes the weight zone could concede to the KV side on demand —
    /// the gauge the pipeline thread publishes each classify
    /// (`PipelineStats::zone_cedeable_bytes`). Feeds the prefill width cap:
    /// the elastic boundary cedes this ground to stuck KV claims
    /// (`request_kv_ground`), so a wave sized against it is admissible even
    /// when little KV ground is standing free. Reads 0 before the first
    /// classify — the cold-start waves are far below any cap that matters.
    pub fn cedeable_span_bytes(&self) -> usize {
        PipelineStats::snapshot(&self.stats).zone_cedeable_bytes
    }

    /// Reset all pipeline telemetry counters to zero.
    pub fn reset_expert_stats(&self) {
        PipelineStats::reset(&self.stats);
    }

    /// Record a profiling span from external callers (e.g. SparseMoeBlock).
    ///
    /// Recorded twice, into two tables that answer two questions. This cache's
    /// own is per-instance and is what the bench harness snapshots per config;
    /// the pipeline profiler is process-wide and is what a running daemon serves
    /// over its API, where an expert-routing cost that appeared in neither the
    /// scheduler's phases nor the kernel spans would otherwise be a hole in the
    /// breakdown exactly where the MoE lives.
    ///
    /// When profiling is disabled, this is an inline no-op.
    #[cfg(feature = "profile")]
    pub fn record_profile(&self, name: &'static str, start: ProfileMark) {
        if let Ok(mut prof) = self.forward_profile.lock() {
            prof.record(name, start);
        }
        crate::models::profile::pipeline_record(name, start);
    }

    /// No-op when profiling is disabled.
    #[cfg(not(feature = "profile"))]
    #[inline(always)]
    pub fn record_profile(&self, _name: &'static str, _start: ProfileMark) {}
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        // Profile report is now collected per-config via snapshot_profiles().
        // Only print at shutdown if un-collected data remains.
        #[cfg(feature = "profile")]
        {
            if let Ok(prof) = self.forward_profile.lock() {
                if prof.has_data() {
                    println!("{}", prof.report("Forward Thread Profile (uncollected)"));
                }
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod warm_sizing_tests {
    use super::{
        warm_sizing_from, CEILING_AVAILABLE, CEILING_HOST_BUDGET, CEILING_NONE, CEILING_PINNABLE,
        PAGEABLE_RESERVE, WARM_TIER_HEADROOM,
    };

    const GIB: u64 = 1024 * 1024 * 1024;
    /// The 3.6-35B's slot: three projections at their aligned offsets.
    const SLOT: usize = 1_933_312;
    /// Its evictable set — 39 unpinned layers of 256.
    const EVICTABLE: usize = 9_984;

    /// A budget generous enough not to be the binder, so a case can isolate one
    /// of the other two ceilings.
    const LOOSE_BUDGET: u64 = 1024 * GIB;

    /// **The regression this exists for.** The 16 GB dev box, sized from the
    /// mid-load trough (12.18 GiB) against the launch reading (20.45 GiB).
    ///
    /// The live figure is depressed by the engine's own mapped checkpoint, and
    /// sizing from it cost 3,030 experts — every one of which then reads the
    /// pack on a miss, for the life of the process.
    #[test]
    fn the_launch_baseline_beats_the_mid_load_trough() {
        let args = |launch: u64| {
            warm_sizing_from(
                SLOT,
                EVICTABLE,
                31 * GIB + GIB / 2,
                12 * GIB + GIB / 5, // 12.18 GiB free mid-load
                launch,
                640 * 1024 * 1024, // embedding + staging already pinned
                27 * GIB,
            )
        };
        let trough = args(12 * GIB + GIB / 5);
        let launch = args(20 * GIB + GIB / 2);
        assert!(
            launch.slots > trough.slots + 3000,
            "launch baseline bought only {} slots over the trough's {}",
            launch.slots,
            trough.slots
        );
        // Availability is the ceiling that is *supposed* to decide on a machine
        // like this — the pinnable reserve is a backstop, not the everyday
        // binder. Both cases are availability-bound; the launch reading simply
        // gives it a truthful number to work from.
        assert_eq!(trough.bound_by, CEILING_AVAILABLE);
        assert_eq!(launch.bound_by, CEILING_AVAILABLE);
    }

    /// **Every ceiling leaves the headroom, including the pinnable one.**
    ///
    /// The regression: on a settled 31.5 GiB box the launch baseline (20.05 GiB)
    /// lifted the availability ceiling above the pinnable one for the first
    /// time, the pinnable ceiling bound, and — having no reserve subtracted from
    /// it — handed the tier the entire pinnable region with nothing left over.
    /// The first forward died on `CUDA_ERROR_OUT_OF_MEMORY`.
    ///
    /// The invariant: whatever binds, total pinned must stay a headroom clear of
    /// the pinnable limit. Exercised on a machine small enough that the pinnable
    /// ceiling is the one that wins.
    #[test]
    fn no_ceiling_may_take_the_whole_pinnable_region() {
        let total = 16 * GIB;
        let already = 640 * 1024 * 1024;
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            total,
            60 * GIB, // free RAM is not the constraint here
            60 * GIB,
            already,
            LOOSE_BUDGET,
        );
        assert_eq!(s.bound_by, CEILING_PINNABLE);
        assert!(
            s.taken_bytes + already + WARM_TIER_HEADROOM + PAGEABLE_RESERVE <= total,
            "tier {:.2} GiB + {:.2} GiB already pinned does not leave the {:.2} GiB headroom \
             and {:.2} GiB pageable reserve inside {:.2} GiB",
            s.taken_bytes as f64 / GIB as f64,
            already as f64 / GIB as f64,
            WARM_TIER_HEADROOM as f64 / GIB as f64,
            PAGEABLE_RESERVE as f64 / GIB as f64,
            total as f64 / GIB as f64,
        );
    }

    /// A machine with no room left over after the pageable reserve takes no
    /// warm tier at all, rather than underflowing into an enormous one.
    #[test]
    fn a_machine_smaller_than_the_pageable_reserve_takes_nothing() {
        for total_gib in [4u64, 8, 10, 12] {
            let s = warm_sizing_from(
                SLOT,
                EVICTABLE,
                total_gib * GIB,
                60 * GIB,
                60 * GIB,
                0,
                LOOSE_BUDGET,
            );
            assert_eq!(
                s.slots, 0,
                "{total_gib} GiB machine has no pinnable room past the reserve"
            );
        }
    }

    /// The same invariant, swept: no combination of machine size, free RAM or
    /// standing pinned bytes may leave the pageable reserve short. A ceiling
    /// that can bind has to leave the same room as its neighbours, so this holds
    /// whichever one wins.
    #[test]
    fn the_pageable_reserve_holds_across_machines() {
        for total_gib in [8u64, 16, 31, 64, 194] {
            for free_gib in [2u64, 8, 30, 120] {
                for pinned_gib in [0u64, 1, 6] {
                    let total = total_gib * GIB;
                    let already = pinned_gib * GIB;
                    let s = warm_sizing_from(
                        SLOT,
                        EVICTABLE,
                        total,
                        free_gib * GIB,
                        free_gib * GIB,
                        already,
                        LOOSE_BUDGET,
                    );
                    // Stated as what the tier may take, not as what total pinned
                    // must be: a process that has *already* pinned past the
                    // limit is not something this function can undo, and the
                    // only correct answer there is to take nothing — which
                    // `saturating_sub` gives.
                    assert!(
                        s.taken_bytes
                            <= total
                                .saturating_sub(PAGEABLE_RESERVE)
                                .saturating_sub(already)
                                .saturating_sub(WARM_TIER_HEADROOM),
                        "{total_gib} GiB machine, {free_gib} GiB free, {pinned_gib} GiB pinned: \
                         tier took {} bytes",
                        s.taken_bytes
                    );
                }
            }
        }
    }

    /// A machine that gained free RAM since launch is sized on the larger, newer
    /// figure — the baseline is a floor on optimism, not a cap.
    #[test]
    fn a_machine_that_freed_ram_is_not_held_to_its_launch_reading() {
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            64 * GIB,
            40 * GIB, // plenty free now
            8 * GIB,  // but the process launched on a busy machine
            0,
            LOOSE_BUDGET,
        );
        assert_eq!(s.launch_ram, 40 * GIB);
        assert_eq!(s.available_less_headroom, 40 * GIB - WARM_TIER_HEADROOM);
    }

    /// **Pinned bytes are subtracted exactly once, whichever reading wins.**
    ///
    /// The live figure is taken after the process page-locked them, so it
    /// already excludes them; the launch figure predates them and does not.
    /// Subtracting from the winner double-counts on every machine where the live
    /// reading is larger — the case the `max` exists for. The two tests either
    /// side of this one are both blind to it: one passes `already_pinned = 0`,
    /// the other makes launch and live equal.
    #[test]
    fn pinned_bytes_are_not_double_subtracted_when_the_live_reading_wins() {
        let already = 4 * GIB;
        // Live is the larger *and* already nets out `already`; launch predates
        // it. Normalised, both describe the same 40 GiB of usable ground.
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            64 * GIB,
            40 * GIB,
            44 * GIB,
            already,
            LOOSE_BUDGET,
        );
        assert_eq!(
            s.available_less_headroom,
            40 * GIB - WARM_TIER_HEADROOM,
            "the pinned bytes were counted twice"
        );
    }

    /// **Already-pinned bytes come out of the LAUNCH reading, which predates
    /// them — never out of the live one, which already excludes them.**
    ///
    /// This test used to hold launch and live equal and demand the ceiling move
    /// one-for-one, which is the double-count: with both at 40 GiB and 4 GiB
    /// pinned, the live figure says 40 GiB is free *now, after* the pinning, and
    /// subtracting again invents a shortage. The pinnable cap is different and
    /// does subtract, because it is derived from `total_ram` — a constant, not a
    /// reading, so nothing has netted the pinned bytes out of it.
    #[test]
    fn what_is_already_pinned_is_subtracted_from_the_launch_reading() {
        // Live is stale (smaller), so the launch reading wins and must pay.
        let none = warm_sizing_from(
            SLOT,
            EVICTABLE,
            64 * GIB,
            8 * GIB,
            40 * GIB,
            0,
            LOOSE_BUDGET,
        );
        let some = warm_sizing_from(
            SLOT,
            EVICTABLE,
            64 * GIB,
            8 * GIB,
            40 * GIB,
            4 * GIB,
            LOOSE_BUDGET,
        );
        assert_eq!(
            none.available_less_headroom - some.available_less_headroom,
            4 * GIB,
            "pinned bytes must move a launch-derived ceiling one-for-one"
        );
        // The pinnable cap always subtracts: `total_ram` is a constant.
        assert_eq!(none.pinnable_cap - some.pinnable_cap, 4 * GIB);
    }

    /// Each ceiling binds when it is the lowest, and says so by name — the whole
    /// point of reporting `bound_by` rather than three numbers to compare.
    #[test]
    fn the_lowest_ceiling_binds_and_is_named() {
        // Host budget lowest.
        let s = warm_sizing_from(SLOT, EVICTABLE, 64 * GIB, 60 * GIB, 60 * GIB, 0, 5 * GIB);
        assert_eq!(s.bound_by, CEILING_HOST_BUDGET);
        assert_eq!(s.slots, (5 * GIB / SLOT as u64) as usize);

        // Availability lowest.
        let s = warm_sizing_from(SLOT, EVICTABLE, 64 * GIB, 6 * GIB, 6 * GIB, 0, LOOSE_BUDGET);
        assert_eq!(s.bound_by, CEILING_AVAILABLE);

        // Pinnable half lowest — and it leaves the headroom like the others, so
        // an 8 GiB machine offers a 4 GiB pinnable region of which the tier may
        // take 1 GiB.
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            8 * GIB,
            60 * GIB,
            60 * GIB,
            0,
            LOOSE_BUDGET,
        );
        assert_eq!(s.bound_by, CEILING_PINNABLE);
        assert_eq!(
            s.slots,
            ((4 * GIB - WARM_TIER_HEADROOM) / SLOT as u64) as usize
        );

        // Nothing binds: the tier covers every evictable expert.
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            512 * GIB,
            400 * GIB,
            400 * GIB,
            0,
            LOOSE_BUDGET,
        );
        assert_eq!(s.bound_by, CEILING_NONE);
        assert_eq!(s.slots, EVICTABLE);
    }

    /// A machine with less free than the headroom asks for takes no tier rather
    /// than underflowing into an enormous one.
    #[test]
    fn a_machine_below_the_headroom_takes_nothing() {
        let s = warm_sizing_from(SLOT, EVICTABLE, 8 * GIB, GIB, GIB, 0, LOOSE_BUDGET);
        assert_eq!(s.available_less_headroom, 0);
        assert_eq!(s.slots, 0);
        assert_eq!(s.taken_bytes, 0);

        // Same for a process already pinning more than the baseline.
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            64 * GIB,
            40 * GIB,
            40 * GIB,
            60 * GIB,
            LOOSE_BUDGET,
        );
        assert_eq!(s.slots, 0);
    }

    /// The reported bytes are what the slots actually cost, never the ceiling
    /// they were cut from — a report that rounded up would hide a shortfall.
    #[test]
    fn taken_bytes_follows_the_slot_count() {
        let s = warm_sizing_from(SLOT, EVICTABLE, 31 * GIB, 20 * GIB, 20 * GIB, 0, 27 * GIB);
        assert_eq!(s.taken_bytes, (s.slots * SLOT) as u64);
        assert!(s.taken_bytes <= s.available_less_headroom.min(s.pinnable_cap));
    }
}
