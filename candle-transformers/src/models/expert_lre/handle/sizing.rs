//! How much host RAM the warm expert tier may pin, worked out from the machine's
//! free memory and the ceiling on page-locked memory.
//!
//! The tier is sized from three ceilings, and the lowest one binds and is named
//! in the report so the reader does not have to infer it. The machine's own
//! budget and the memory free at launch bound it on availability, and the launch
//! figure has the process's own heap taken out, since that heap is not
//! droppable. The third ceiling caps page-locked memory at half the machine,
//! because availability counts droppable page cache and so cannot see the limit
//! on pinned memory. Every ceiling leaves the same headroom for what the process
//! needs after the tier, because pinned pages cannot be reclaimed under pressure.
//! Warm KV is outside this partition and has a pageable floor of its own.

use super::*;

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
/// **3 GiB, sized by the hungriest gate, measured on the 16 GB box** (31.5 GiB
/// host). With the wave's transients on the span's bump tiers, Flash-Next's own
/// transient after pinning is ~1.06 GiB (free RAM low-water 2.94 GiB under a
/// 4 GiB headroom), and its Q3_KO rung — the tier under a third of the experts,
/// prefill bound by the bytes read off the drive — gains from every slot: 3 GiB
/// took the tier to 8,932 experts and BF16×8 prefill to 570.5 t/s (528.4 at
/// 4 GiB), low-water 2.17 GiB. Qwen3.6-35B-A3B's transient is ~2 GiB larger: at
/// 2 GiB its free RAM bottomed at **0.03 GiB**, at 3 GiB at 0.88 GiB. The history
/// below is what the figure was before, when the tier was past its knee and the
/// transient was larger.
///
/// **The earlier figure was 4 GiB, measured rather than reasoned.** The run's
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
/// point was 3.57 GiB. Warm KV does not come out of this: it has a pageable
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
/// against live free RAM instead (`docs/expert_cache_design.md` §6.3).
pub const WARM_TIER_HEADROOM: u64 = 3 * 1024 * 1024 * 1024;

/// How many warm slots to ask for: **every expert the machine will actually
/// give room for.**
///
/// The target is the whole model. A miss that reaches the cold tier is an NVMe
/// read by the stager that the expert GEMM's worker blocks spin on, where a warm
/// miss is a pinned read nothing waits for — so the warm tier is not an
/// optimisation over the pack, it is what keeps the pack off the critical path.
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

pub(super) static WARM_SIZING: Mutex<Option<WarmTierSizing>> = Mutex::new(None);

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
pub(super) fn warm_sizing_from(
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

/// What was free at launch, `launch`, less what this process now holds resident
/// that it will not give back and has not pinned: `own_resident`, its private
/// working set, less `pinned`, which [`warm_sizing_from`] takes out itself.
///
/// **The launch figure's gap is not all droppable.** It stands for the mapped
/// checkpoint's pages, which the warm tier never competed with — but the
/// process's own heap is in the gap too, and that stays. On WhiteWorld (64 GB,
/// 2026-11-17) zend's opened substrate held ~31 GB resident; sized from the
/// launch figure, the tier pinned 30 GB with 18.9 GB free, the process's
/// working set was paged out to 0.02 GB, the pagefile reached 47 GB, and the
/// boot ran past keep's ten-minute trial. Taken out here, the same machine
/// sizes the tier to what it has.
pub(super) fn launch_less_own(launch: u64, own_resident: u64, pinned: u64) -> u64 {
    launch.saturating_sub(own_resident.saturating_sub(pinned))
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
    let pinned = candle::vram::host_pinned_bytes();
    let sizing = warm_sizing_from(
        stride,
        total_experts,
        total_ram,
        available,
        launch_less_own(
            candle::vram::launch_available_ram().unwrap_or(available),
            candle::vram::process_private_resident().unwrap_or(pinned),
            pinned,
        ),
        pinned,
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
