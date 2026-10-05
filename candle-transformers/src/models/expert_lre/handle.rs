//! Public API — the [`ExpertCache`] handle.
//!
//! [`ExpertCache`] is the main entry point for the expert pipeline. It owns the
//! device-side expert forward ([`ExpertCache::forward_routed`], see `dispatch`)
//! and the channels to the two host threads: the pipeline thread, which keeps
//! VRAM residency in step with routing, and the stager, which reads cold
//! experts into the pad. The forward thread waits on neither: it enqueues each
//! MoE layer, tells both the layer's routing, and moves on; the GPU waits only
//! on a cold expert the stager is still reading.
//!
//! There is no host-side expert path, and no CPU one: a build without `cuda`
//! cannot construct the cache.

use super::cache::{minimum_resident_slots, pinned_layer_count, ExpertCacheInner};
#[cfg(feature = "cuda")]
use super::dispatch::Dispatch;
#[cfg(feature = "cuda")]
use super::live_table::LiveTable;
#[cfg(feature = "cuda")]
use super::pack::{
    open_or_create, repack_fingerprint, LayerSpansInput, PackIdentity, PackSource, PackSpec,
    RecordLayout,
};
#[cfg(feature = "cuda")]
use super::pad::Pad;
#[cfg(feature = "cuda")]
use super::pinned::{stratified_membership, ExpertResidency};
#[cfg(feature = "cuda")]
use super::pipeline::{spawn_pipeline_thread, PipelineState};
#[cfg(feature = "cuda")]
use super::promo::PromotionRing;
#[cfg(feature = "cuda")]
use super::residency::Residency;
#[cfg(feature = "cuda")]
use super::slot_image::{row_tile_bytes_for, slot_bytes_for, slot_offsets};
#[cfg(feature = "cuda")]
use super::stager::{spawn_stager, StagerCtx, StagerMsg};
#[cfg(feature = "cuda")]
use super::startup::{
    startup_from_pack, startup_pinned_prefix, startup_repack, StartupRing, StartupTargets,
    PACK_TAIL_STEPS,
};
use super::types::{MmapExpertRef, PipelineMessage, PipelineStats};
#[cfg(feature = "cuda")]
use super::warm_tier::WarmTier;
use crate::models::profile::ProfileSnapshot;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::Q8a128Operand;
#[cfg(feature = "cuda")]
use candle::quantized::decode_rows::DecodeRows;
use candle::quantized::Int8Mode;
#[cfg(feature = "cuda")]
use candle::{DType, LiveTensor};
use candle::{Device, Result};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::WaveGeneration;
use candle_nn::kv_cache::WeightZone;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

/// Pad slots an all-resident cache takes: nothing is ever cold there, so the
/// pad serves only as the startup fill's landing ring.
#[cfg(feature = "cuda")]
const STARTUP_RING_SLOTS: usize = 64;

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
/// which this gate barely exercises and a daemon workload does.
///
/// **Smaller was measured again and is slower, not faster**, on Flash-Next at
/// 16 GB with 21.9 GiB free at launch. At 3 GiB the warm tier grew 13,508 →
/// 14,283 experts and pack reads fell ~9%, but free RAM bottomed at 2.05 GiB and
/// the run paid for it: decode 13.8 / 67.6 / 26.8 → 13.5 / 64.7 / 22.8 t/s
/// (BF16×1 / BF16×8 / C5×2), prefill 269.5 → 256.0 on C5. At 4 GiB the low
/// point is 3.57 GiB. Warm KV does not come out of this: it has a pageable
/// floor of its own (`vram::KV_WARM_FLOOR`) that the OS pages rather than
/// refusing.
///
/// **Deliberately NOT raised for calibration's sake.** A from-empty full tool
/// calibration (`zend::session`'s "Calibrating sections", thousands of
/// `kv_lossless`-pinned cases) needs several GiB more transient room than this
/// figure leaves and starved on a 31.5 GiB box (`memory allocation of 16777216
/// bytes failed` 15–51 % through the corpus, `vram::available_low_water` reading
/// single-digit MiB). Raising this constant to cover it was tried and reverted:
/// `the_pinned_expert_tier_fits_the_machine` — this crate's own regression test
/// for the everyday case, a 31.5 GiB box with 20 GiB free at launch — measured
/// the tier dropping to exactly the new headroom's shortfall,
/// "too tight to be worth the pack-file misses it avoids". This headroom taxes
/// every boot; calibration's much larger, one-time transient is the exception,
/// not the common case, and belongs in the exception's own code —
/// `zend::session`'s calibration loop backs its concurrency window off directly
/// against live free RAM instead. See `zend_run_iteration_traps.md` §4.
pub const WARM_TIER_HEADROOM: u64 = 4 * 1024 * 1024 * 1024;

use candle::vram::PAGEABLE_RESERVE;

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
/// `cuMemAllocHost` remains the authority for the pinned part — `WarmPool::new`
/// steps the request down by 512 MiB on refusal, and whatever it cannot pin
/// the warm tier holds pageable (`WarmTier`) — but every refusal costs a round
/// trip, so the first ask should be one that can succeed.
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
    /// Ceiling 1: this tier's slice of the host partition — the tier pool left
    /// once the mmap'd weights, the pageable reserve and the OS floor are taken
    /// out, less the warm KV tier's share of it.
    pub expert_pinned_budget: u64,
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
pub const CEILING_HOST_BUDGET: &str = "expert share of the host tier pool";
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
    expert_pinned_budget: u64,
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
    let affordable = expert_pinned_budget
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
        expert_pinned_budget,
        available_less_headroom,
        pinnable_cap,
        bound_by: if slots == total_experts {
            CEILING_NONE
        } else if affordable == expert_pinned_budget {
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

/// How many `stride`-byte pinned slots this host can afford, wanting
/// `total_slots` of them.
///
/// `pub(crate)` and named for slots rather than experts because a **dense**
/// model's warm tier asks the identical question of the identical machine — its
/// slots hold layers instead of experts, and a checkpoint is one or the other,
/// so the two tiers never coexist and never compete. `layer_stream` calls this;
/// duplicating the three ceilings for it would be a second place for the
/// page-lock cap to be forgotten, which is the one of the three that no
/// availability reading can see.
pub(crate) fn warm_slots_for(stride: usize, total_slots: usize) -> usize {
    let total_experts = total_slots;
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
        // This tier's OWN slice of the host partition. It used to be handed the
        // KV tier's budget as a ceiling, which never bound: this runs before the
        // pool is pinned, so that figure was computed against a machine with
        // nothing pinned in it and came out near the size of the whole box.
        budget.expert_pinned_budget_bytes,
    );
    let slots = sizing.slots;
    let bound_by = sizing.bound_by;
    if let Ok(mut g) = WARM_SIZING.lock() {
        *g = Some(sizing);
    }
    tracing::info!(
        target: "candle_transformers::expert_lre",
        total_gib = total_ram as f64 / 1e9,
        available_gib = available as f64 / 1e9,
        // The whole partition, so a run's log says where every byte went and
        // which of the three ceilings actually bound the tier — reading a `take`
        // without them leaves the reader guessing at the arithmetic.
        tier_pool_gib = budget.tier_pool_bytes as f64 / 1e9,
        expert_budget_gib = budget.expert_pinned_budget_bytes as f64 / 1e9,
        kv_warm_budget_gib = budget.kv_warm_budget_bytes as f64 / 1e9,
        pageable_reserve_gib = budget.pageable_reserve_bytes as f64 / 1e9,
        weights_gib = budget.weights_reserved_bytes as f64 / 1e9,
        take_gib = (slots * stride) as f64 / 1e9,
        bound_by,
        slots,
        of = total_experts,
        "warm tier: sized from the host partition"
    );
    slots
}

// ============================================================================
// ExpertCache — the public handle
// ============================================================================

/// Global expert cache / pipeline handle.
///
/// Shared across all MoE blocks via `Arc<ExpertCache>`. A background pipeline
/// thread owns all mutable cache state (`ExpertCacheInner`, slots, eviction
/// scores, the copy stream) with `&mut self`; the forward thread reaches it
/// only through the channel and the pass lock (`dispatch::PassState`).
pub struct ExpertCache {
    /// Channel to the pipeline thread.
    #[cfg(feature = "cuda")]
    tx: mpsc::SyncSender<PipelineMessage>,
    /// Channel to the stager.
    #[cfg(feature = "cuda")]
    stager: mpsc::Sender<StagerMsg>,
    /// The device-side expert forward and everything it shares with the host
    /// threads: the live table, the summary ring, the abort word.
    #[cfg(feature = "cuda")]
    dispatch: Dispatch,
    /// Set when the pipeline thread or the stager has exited (normally or by
    /// panic). Their guards raise the abort word as they go, so every waiting
    /// worker traps instead of waiting on an expert that will never come.
    #[cfg(feature = "cuda")]
    pipeline_dead: Arc<AtomicBool>,
    #[cfg(feature = "cuda")]
    stager_dead: Arc<AtomicBool>,
    /// The pinned tiers the live table's entries point into — held here so
    /// they outlive every reader, whichever thread exits first.
    #[cfg(feature = "cuda")]
    _warm: Arc<WarmTier>,
    #[cfg(feature = "cuda")]
    _pad: Arc<Pad>,
    /// Shared telemetry counters (always-on).
    stats: Arc<Mutex<PipelineStats>>,
}

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
    /// Experts each token routes to (top-k) — sizes the bucketize workspace and
    /// is checked against the router's bound.
    pub experts_used: usize,
    /// The checkpoint the experts come from — names the pack and identifies it.
    pub gguf_path: &'a std::path::Path,
    /// Where a persistent pack lives, or `None` for a temp file that is
    /// unlinked as soon as it is open and costs a repack every boot.
    pub expert_pack_dir: Option<&'a std::path::Path>,
    pub progress: Option<&'a dyn Fn(usize, usize)>,
    pub int8mode: Int8Mode,
    /// Mapped bytes outside the experts that host RAM never serves after load —
    /// read through a bounded cache of the model's own (net of that cache's
    /// RAM), or uploaded to the device once and never read from the file again.
    /// Flash-Next's n-gram table is 54 GB of the mapping read through a 2 GiB row
    /// cache, and its dense stack is 5.2 GiB living on the card; left counted as
    /// live weight, the host budget reserves them as page cache out of the warm
    /// tier's RAM. `0` for a model whose whole non-expert mapping is paged —
    /// one that reads a weight from the mapping at run time, such as a
    /// host-mapped embedding gather.
    pub offloaded_bytes: u64,
}

impl ExpertCache {
    /// A build without `cuda` has no expert path: the experts run only as the
    /// device-side grouped GEMM over the live pointer table.
    #[cfg(not(feature = "cuda"))]
    pub fn new(_setup: ExpertCacheSetup<'_>) -> Result<Self> {
        candle::bail!(
            "MoE expert cache: the routed experts run only on the device — build with the \
             `cuda` feature"
        )
    }

    /// Create a new expert cache with a background pipeline thread.
    ///
    /// Opens the pack file for this checkpoint — building it by repacking every
    /// expert out of the GGUF if there is not already a matching one — then
    /// fills the warm and hot tiers from it, and builds the live pointer table
    /// over the hot tier. After startup the GGUF's expert regions are never read
    /// again. Requires an actual CUDA device: a cuda-feature build handed a CPU
    /// device fails here rather than later.
    #[cfg(feature = "cuda")]
    pub fn new(setup: ExpertCacheSetup<'_>) -> Result<Self> {
        let ExpertCacheSetup {
            mmap,
            host_refs,
            zone,
            device,
            experts_per_layer,
            experts_used,
            gguf_path,
            expert_pack_dir,
            progress,
            int8mode,
            offloaded_bytes,
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

        // ── The copy stream: every promotion and relocation, in order ──
        let Device::Cuda(cuda_dev_ref) = device else {
            candle::bail!(
                "MoE expert cache: this build has CUDA compiled in but was given {device:?}. \
                 The expert tiers are all device-side or DMA-bound, so there is no CPU path."
            )
        };
        let copy_stream = cuda_dev_ref
            .cuda_context()
            .new_stream()
            .map_err(candle::Error::wrap)?;

        // ── CUDA startup: the pack, then the resident tiers from it ──
        let (pack, warm, pad, residency, layer_geometries, all_resident) =
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
                // The same holds for any other region a bounded cache of the
                // model's own serves (`offloaded_bytes`): its pages are not the
                // page cache's to keep.
                let live_weight_bytes = (mmap.len() as u64)
                    .saturating_sub(expert_source_bytes)
                    .saturating_sub(offloaded_bytes);
                candle::vram::set_weights_mmap(live_weight_bytes);
                tracing::info!(
                    target: "candle_transformers::expert_lre",
                    mapped_gib = mmap.len() as f64 / 1e9,
                    dead_gib = expert_source_bytes as f64 / 1e9,
                    offloaded_gib = offloaded_bytes as f64 / 1e9,
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
                // **The pad before the warm tier, not after.** It is mandatory —
                // it is what makes a cold expert computable — and one layer of
                // slots; the warm tier is elastic and pins up to the driver's
                // page-lock ceiling, so anything pinned after it lands on an
                // exhausted budget. The startup fill borrows the pad as its
                // landing ring before the stager takes it over.
                let pad = Pad::new(
                    if all_resident {
                        STARTUP_RING_SLOTS
                    } else {
                        experts_per_layer.max(STARTUP_RING_SLOTS)
                    },
                    stride,
                )?;
                // **A cache that holds every expert in VRAM wants no warm tier
                // at all.** Nothing is ever evicted in that state, so no routed
                // expert is ever read from a host tier. Sizing it anyway would pin the model's size in host RAM
                // to serve nothing, and pay a full-pack read at startup for it.
                // The warm tier's job is covering **misses**, and the pinned
                // prefix never generates one, so it is sized against the
                // evictable set rather than the model. On the 3.6-35B that is
                // 512 fewer experts to aim at — 943 MiB of pinned host memory
                // that used to be spent on experts no load could ever ask for.
                let pinned = pinned_layer_count(num_moe_layers);
                let evictable = total_experts - pinned * experts_per_layer;
                let mut residency =
                    vec![vec![ExpertResidency::default(); experts_per_layer]; num_moe_layers];
                // The pinned prefix, from the checkpoint, before the warm tier
                // takes the page-lock budget its pageable uploads need.
                startup_pinned_prefix(
                    &mut inner,
                    &mut residency,
                    &geoms,
                    &mmap,
                    &host_refs,
                    cuda_dev,
                    progress,
                )?;
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
                // Pinned as far as the driver grants while keeping its margin,
                // pageable for the rest — see `warm_tier`.
                let mut warm = WarmTier::new(membership.len(), stride);
                // A refusal shortens the draw rather than leaving slots the tier
                // does not have: `ram` must never name a slot outside it.
                let membership = &membership[..membership.len().min(warm.num_slots())];

                // The eviction policy weighs what a reload would cost, so it has
                // to know which experts the warm tier holds before the first
                // victim is chosen.
                inner.set_warm_backed(membership);
                let mut ring = StartupRing::new(&pad);
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
                            &mut ring,
                            num_moe_layers,
                            experts_per_layer,
                            cuda_dev,
                            progress,
                        )?;
                        pack
                    }
                    PackSource::Build(mut writer) => {
                        startup_repack(
                            targets,
                            &mut writer,
                            &mut ring,
                            cuda_dev,
                            progress,
                        )?;
                        // Publishing flushes and `fsync`s the whole pack — tens
                        // of gigabytes, and the single largest thing that used
                        // to happen behind a bar already reading 100%. It is
                        // charged here, where it is paid, against the room
                        // `startup_repack` left for it.
                        let total = num_moe_layers * experts_per_layer;
                        let t_publish = std::time::Instant::now();
                        let pack = writer.finish()?;
                        // Timed and named, because the next line printed used to
                        // be the residency gauge's and the whole flush was read
                        // off the log as the gauge being slow. The gauge is three
                        // arithmetic operations.
                        tracing::info!(
                            target: "candle_transformers::expert_lre",
                            secs = t_publish.elapsed().as_secs_f64(),
                            "expert pack: published (flush + fsync + reopen)"
                        );
                        if let Some(cb) = progress {
                            cb(total + 1, total + PACK_TAIL_STEPS);
                        }
                        pack
                    }
                };

                // The pack's stride is what the geometry said it would be — the
                // buffers above were cut to it before the file was opened.
                debug_assert_eq!(pack.stride(), stride);
                drop(ring);
                (pack, warm, pad, residency, geoms, all_resident)
            } else {
                // Refused above, where the copy stream is created; kept as the
                // match's other arm so the binding stays a plain `let`.
                candle::bail!("MoE expert cache: expected a CUDA device, got {device:?}")
            };

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
            // The last of the tail: the cache is built and reporting itself, so
            // the bar lands on its total here rather than at the last expert.
            if let Some(cb) = progress {
                let total = num_moe_layers * experts_per_layer;
                cb(total + PACK_TAIL_STEPS, total + PACK_TAIL_STEPS);
            }
            if let Ok(mut s) = stats.lock() {
                s.resident_vram_bytes = seeded;
                s.warm_slots = warm.num_slots();
                s.warm_paged_slots = warm.paged_slots();
                s.total_experts = num_moe_layers * experts_per_layer;
                s.moe_layers = num_moe_layers;
                // **The zone's shape, seeded here and not left to the first classify.**
                //
                // These four are what `WeightPlan::from_stats` needs, and it refuses the
                // whole gauge set if any reads zero — correctly, since a zero slot size
                // makes a routed expert look free. They used to be written only by
                // `classify_and_load`, which on an all-resident cache does no loading
                // worth the name: nothing streams, so nothing refreshed them, so they
                // stayed at zero for the process lifetime.
                //
                // The consequence was an inversion. With no weight plan the scheduler's
                // rate planner cannot be armed, and admission falls back to one prefill
                // per pass — so a card *large enough to hold the whole checkpoint* ran
                // waves one row wide, while a card small enough to stream experts
                // published gauges, planned, and batched. Measured on the 30B-A3B at 72
                // GiB: `decode seqs avg=1.0 max=1` and 32 t/s against a batched ceiling
                // of 518.
                //
                // Every term is known here — the zone is carved before this point and
                // `slot_bytes` is the same figure the resident gauge above is a multiple
                // of — so there was never a reason to wait for a classify.
                let zone_slot_bytes = inner.zone.slot_bytes();
                s.expert_slot_bytes = zone_slot_bytes;
                s.zone_bytes = inner.zone.capacity() * zone_slot_bytes;
                s.zone_min_bytes = inner.zone.min_capacity() * zone_slot_bytes;
                s.zone_max_bytes = inner.zone.limit() * zone_slot_bytes;
            }
        }

        // ── The live table, over where the fill just put every expert ──
        //
        // Built for every cache, all-resident or not: there is one expert path,
        // and the table is how the device finds a weight. Every condition that
        // path depends on is checked here and refused with its reason.
        let table = Arc::new(LiveTable::new(
            cuda_dev_ref,
            &layer_geometries,
            experts_per_layer,
            experts_used,
        )?);
        let mut places = Residency::new(table.clone());
        for (row, layer) in residency.iter().enumerate() {
            for (expert, res) in layer.iter().enumerate() {
                if let Some(slot) = res.vram {
                    places.set_vram(row, expert, Some((slot, inner.slot_base(slot))));
                }
                if let Some(ws) = res.ram {
                    places.set_warm(row, expert, Some((ws, warm.pinned_addr(ws))));
                }
            }
        }
        places.publish_all();
        let places = Arc::new(Mutex::new(places));
        // Four layers of ring indices: up to two layers' worth stocked with
        // free slots (a cold layer, with the next one's refill still behind
        // the GPU), the rest room for the slots the GPU has taken and this
        // thread has not yet collected. A cache holding every expert misses
        // nothing.
        let promo = if all_resident {
            None
        } else {
            Some(Arc::new(PromotionRing::new(
                4 * experts_per_layer,
                num_moe_layers,
                experts_per_layer,
            )?))
        };
        let dispatch = Dispatch::new(
            cuda_dev_ref,
            table,
            [warm.pinned_range(), pad.range()],
            row_tile_bytes_for(&layer_geometries),
            experts_used,
            promo.clone(),
        )?;
        if let Ok(mut s) = stats.lock() {
            s.pad_slots = pad.num_slots();
        }

        let pack = Arc::new(pack);
        let warm = Arc::new(warm);
        let pad = Arc::new(pad);
        let stager_dead = Arc::new(AtomicBool::new(false));
        let stager = spawn_stager(
            StagerCtx {
                pack,
                warm: warm.clone(),
                pad: pad.clone(),
                residency: places.clone(),
                clock: dispatch.clock.clone(),
                ring: dispatch.ring.clone(),
                abort: dispatch.abort.clone(),
                consumed: dispatch.staged.clone(),
                stats: stats.clone(),
                rows: num_moe_layers,
                n_experts: experts_per_layer,
            },
            stager_dead.clone(),
        )?;

        let state = PipelineState::new(
            inner,
            device.clone(),
            copy_stream,
            places,
            dispatch.clock.clone(),
            dispatch.ring.clone(),
            dispatch.abort.clone(),
            dispatch.pass.clone(),
            dispatch.served.clone(),
            stager.clone(),
            Arc::new(layer_geometries),
            all_resident,
            promo,
            stats.clone(),
        )?;
        let pipeline_dead = Arc::new(AtomicBool::new(false));
        let tx = spawn_pipeline_thread(state, pipeline_dead.clone());

        Ok(Self {
            tx,
            stager,
            dispatch,
            pipeline_dead,
            stager_dead,
            _warm: warm,
            _pad: pad,
            stats,
        })
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

    /// The routed experts of one MoE layer, on the device — see `dispatch`.
    ///
    /// `acts` is the layer's q8a128 activation `[n_tokens, hidden]`; `weights`
    /// and `indices` are the router's `[n_tokens, k]` output (f32, u32), still
    /// on the device. Returns the routed sum `[n_tokens, hidden]` at
    /// `out_dtype`. The tokens in `decode` are decode rows, which the residency
    /// scoring weights apart from prompt rows.
    ///
    /// Never waits on the host: the layer is enqueued, its routing is handed to
    /// the pipeline thread and the stager, and the call returns. The GPU waits
    /// only in an expert GEMM's worker blocks, on a cold expert the stager is
    /// still reading.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_routed<'w>(
        &self,
        acts: Q8a128Operand<'w>,
        weights: &LiveTensor<'_>,
        indices: &LiveTensor<'_>,
        row: usize,
        decode: &DecodeRows,
        out_dtype: DType,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        self.dispatch.forward(
            &self.tx,
            &self.stager,
            acts,
            weights,
            indices,
            row,
            decode,
            out_dtype,
            wave,
        )
    }

    /// Whether the pipeline thread or the stager has exited. Its guard raised
    /// the abort word as it went, so any expert layer enqueued since traps on
    /// the device and the next synchronize reports it.
    #[cfg(feature = "cuda")]
    pub fn pipeline_dead(&self) -> bool {
        self.pipeline_dead.load(Ordering::Acquire) || self.stager_dead.load(Ordering::Acquire)
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
    /// Zero is an ordinary answer: the zone may already sit at its floor, a
    /// wave may be live, or an expert layer the forward thread has begun may
    /// still be unserved — the boundary moves only with none in flight. The
    /// caller decides whether to retry on that basis rather than spinning on a
    /// claim that cannot succeed.
    ///
    /// Blocks on the pipeline thread's reply — it owns the cache state and is
    /// the only place a boundary move is safe.
    pub fn request_kv_ground(&self, regions: usize) -> u64 {
        if regions == 0 {
            return 0;
        }
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        if !self.send(PipelineMessage::RenegotiateBoundary {
            regions,
            response_tx,
        }) {
            return 0;
        }
        response_rx.recv().unwrap_or(0)
    }

    /// Send to the pipeline thread; `false` if it is gone (or, in a build
    /// without `cuda`, never existed).
    fn send(&self, msg: PipelineMessage) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.tx.send(msg).is_ok()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = msg;
            false
        }
    }

    /// Wait until the pipeline thread has processed every message sent before
    /// this one — every routed layer included — so whatever it writes reads
    /// complete. Returns at once if the thread is gone.
    fn settle(&self) {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        if self.send(PipelineMessage::Settle { response_tx }) {
            let _ = response_rx.recv();
        }
    }

    /// The other direction: take back KV regions that are standing free.
    ///
    /// **Call this only between forwards.** Moving the boundary evicts and
    /// relocates expert slots, and a wave in flight may be reading either, so
    /// `set_weight_floor` refuses while a wave generation is open on the span.
    ///
    /// This used to be driven from the pipeline thread's `post_compute`, which
    /// runs the instant a MoE layer's work is answered — with the forward thread
    /// still inside `ffn_residual` holding that layer's FFN wave guard. So it was
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
        // **Sweep before asking, because the answer is computed from `live`.**
        //
        // `spare_regions` offers the weight side what occupancy says is free, and
        // an unswept arena is counted neither free nor spare. A region whose
        // arena went chunk-empty several waves ago is still `live` until
        // something sweeps it, so without this the negotiation is answered
        // against ground that came free some time ago and nobody has noticed.
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
        if !self.send(PipelineMessage::RenegotiateBoundary {
            regions: 0,
            response_tx,
        }) {
            return 0;
        }
        response_rx.recv().unwrap_or(0)
    }

    /// Snapshot and reset the pipeline thread's profile accumulator (`pipe_*`
    /// spans), and — in a profile build — the expert GEMMs' worker counters:
    /// `moe:worker copy` (summed worker time copying remote experts into
    /// scratch, one count per item) and `moe:worker cold wait` (summed worker
    /// time waiting on the stager). The worker sums are block-time across the
    /// `WORKERS` blocks of a launch, not wall time.
    pub fn snapshot_profiles(&self) -> ProfileSnapshot {
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);
        #[allow(unused_mut)]
        let mut snap = if self.send(PipelineMessage::SnapshotProfile {
            response_tx: resp_tx,
        }) {
            resp_rx.recv().unwrap_or_default()
        } else {
            ProfileSnapshot::default()
        };
        #[cfg(all(feature = "cuda", feature = "profile"))]
        match self.dispatch.drain_worker_counters() {
            Ok(c) => {
                snap.entries
                    .push(("moe:worker copy".into(), c.copy_ns as f64 / 1e6, c.items));
                snap.entries.push((
                    "moe:worker cold wait".into(),
                    c.cold_wait_ns as f64 / 1e6,
                    c.items,
                ));
                tracing::info!(
                    target: "candle_transformers::expert_lre",
                    items = c.items,
                    launches = c.launches,
                    gib = c.bytes as f64 / (1u64 << 30) as f64,
                    copy_ms = c.copy_ns as f64 / 1e6,
                    cold_wait_ms = c.cold_wait_ns as f64 / 1e6,
                    "expert workers since the last snapshot"
                );
            }
            Err(e) => tracing::warn!("expert worker counters unreadable: {e}"),
        }
        snap
    }

    /// The pipeline telemetry counters, complete for every routed layer sent
    /// before the call.
    ///
    /// The pipeline thread serves routed layers concurrently with the forward,
    /// so a counter read the moment a forward returns could still be missing
    /// its last layers; this settles the thread first — one channel round
    /// trip, which after the sampler's synchronize finds the thread idle.
    pub fn expert_stats(&self) -> PipelineStats {
        self.settle();
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
        self.settle();
        PipelineStats::reset(&self.stats);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod warm_sizing_tests {
    use super::{
        warm_sizing_from, CEILING_AVAILABLE, CEILING_HOST_BUDGET, CEILING_NONE, CEILING_PINNABLE,
        PAGEABLE_RESERVE, WARM_TIER_HEADROOM,
    };

    const GIB: u64 = 1024 * 1024 * 1024;

    /// Bytes as GiB, so a failure message reads in the units the budget is
    /// reasoned about in rather than eleven digits.
    fn gib(b: u64) -> f64 {
        b as f64 / GIB as f64
    }
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

    /// **What is page-locked must fit the machine beside everything it owes.**
    ///
    /// The failure this pins, measured on the 16 GB box during a tool
    /// calibration: this tier page-locked 13.5 GiB at launch, the warm KV tier
    /// then grew to 7.1 GiB against a budget that believed it had 14.5, and the
    /// host reached 0.9 GiB free while paging at 1,302 pages/sec — two tiers
    /// sized against snapshots that did not contain each other.
    ///
    /// Warm KV is now a fixed pageable floor outside the partition (the OS pages
    /// it rather than refusing), so the sum that must hold is the pinned one:
    /// this tier, what else is pinned, the weights and the pageable reserve.
    #[test]
    fn the_pinned_expert_tier_fits_the_machine() {
        let total = 31 * GIB + GIB / 2;
        let weights = 2 * GIB + GIB / 2;
        let budget = candle::vram::host_ram_budget_from(total, 0, weights, 30, 1024 * 1024 * 1024);
        let already = 640 * 1024 * 1024;
        let s = warm_sizing_from(
            SLOT,
            EVICTABLE,
            total,
            // A machine with plenty free at launch — the case that let the old
            // sizing take everything above a fixed 4 GiB guess.
            20 * GIB,
            20 * GIB,
            already,
            budget.expert_pinned_budget_bytes,
        );
        let committed = s.taken_bytes
            + budget.weights_reserved_bytes
            + already
            + candle::vram::PAGEABLE_RESERVE;
        assert!(
            committed <= total,
            "expert tier {:.2} + weights {:.2} + pinned {:.2} + \
             pageable reserve {:.2} = {:.2} GiB on a {:.2} GiB machine",
            gib(s.taken_bytes),
            gib(budget.weights_reserved_bytes),
            gib(already),
            gib(candle::vram::PAGEABLE_RESERVE),
            gib(committed),
            gib(total),
        );
        // And the tier must still be worth having — a partition that fixes the
        // over-commit by starving the cache would pass the line above.
        assert!(
            s.taken_bytes > 8 * GIB,
            "the expert tier got only {:.2} GiB; the split is too tight to be \
             worth the pack-file misses it avoids",
            gib(s.taken_bytes),
        );
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
