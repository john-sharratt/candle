//! Decides VRAM pressure from the count of free KV regions, with a wider
//! setpoint for loading KV than for decoding it.
//!
//! Both phases read the same exact counter of free regions and differ only in
//! how many they insist on. Loading brings KV in before attention reads it, and
//! a wide ragged forward claims regions fast, so its setpoint is wide enough
//! that a seal never finds the free list empty. Decode grows KV by about one
//! chunk per sequence per 32 steps, so its thinner setpoint keeps the most KV
//! resident. Setpoints scale with the span, with floors so a small card does not
//! stall mid-seal, and relief frees a little past them.

use super::*;

/// Free KV regions kept in hand before [`Scheduler::vram_under_pressure_for`]
/// calls it pressure, as a divisor of the reservation's KV side plus an absolute
/// floor in regions. This is §3.8's setpoint.
///
/// It replaced a band of *bytes* derived from the driver — headroom held against
/// a wide forward's transient activation peak. That quantity is no longer the KV
/// side's business: transients come from the reservation's other end (§3.6), and
/// what a seal pass needs is simply somewhere to put its chunks. So the setpoint
/// asks the only question that remains, and asks it of an exact counter: are
/// there enough free regions to absorb the work already admitted?
///
/// Scaled to the span rather than fixed, so the same numbers hold on a 3.6 GiB
/// KV side and on the workstation's; the floors are what keeps a small card from
/// setting a setpoint of two regions and stalling mid-seal. Relief frees
/// [`RELIEF_OVERSHOOT_REGIONS`] past it.
///
/// **The weight side is told this target** (`set_kv_free_target`, at scheduler
/// start), and its growth leaves the relieved count free. Without that the two
/// sides of the boundary traded the same regions every decode step — see
/// [`FreeRegionTarget`].
pub(in super::super) const LOAD_TARGET: FreeRegionTarget =
    FreeRegionTarget::new(8, 24, RELIEF_OVERSHOOT_REGIONS);

/// Decode's setpoint is half of load's: a decode step advances one token per
/// sequence, so KV grows by ~one chunk per sequence per 32 steps — orders of
/// magnitude slower than a prefill's upload, and the whole point of unbounded
/// context is to leave KV resident rather than evict it defensively.
pub(super) const DECODE_TARGET: FreeRegionTarget =
    FreeRegionTarget::new(16, 8, RELIEF_OVERSHOOT_REGIONS);

/// The region quantum in bytes.
pub(super) fn region_bytes() -> u64 {
    candle_nn::kv_cache::REGION_BYTES as u64
}

/// The free-region target `phase` defends.
pub(super) fn phase_target(phase: VramPhase) -> FreeRegionTarget {
    match phase {
        VramPhase::Load => LOAD_TARGET,
        VramPhase::Decode => DECODE_TARGET,
    }
}

/// The free-region setpoint for `phase`, in regions, given a KV side of
/// `total` regions. Pure — unit-tested in isolation.
pub(super) fn setpoint_regions(phase: VramPhase, total: usize) -> usize {
    phase_target(phase).setpoint(total)
}

/// The phase a VRAM pressure decision is made in. Both phases read the same
/// exact counter — free regions — and differ only in how many they insist on:
///
/// - [`Load`](VramPhase::Load) — bringing KV into VRAM *before* attention
///   (prefill upload, section/scope ingest, warm→hot elevation). A wide ragged
///   forward claims regions fast, so the setpoint is wide enough that a seal
///   pass never finds the free list empty mid-wave.
/// - [`Decode`](VramPhase::Decode) — one token per sequence per step, so KV
///   grows slowly and predictably. A thin setpoint keeps the maximum KV
///   resident, which is the whole point of unbounded context.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum VramPhase {
    Load,
    Decode,
}

/// Regions the relief sequence frees past the setpoint, so a pass that just
/// clears pressure does not re-trip on the very next wave. Eviction is bulk and
/// coarse by nature — one turn's hot copy spans many chunks — so overshooting
/// deliberately is cheaper than nibbling every wave, which is what caused the
/// reload churn the old watermark ladder was built to damp.
pub(super) const RELIEF_OVERSHOOT_REGIONS: usize = 8;

impl Scheduler {
    /// What the card can actually deliver to admission right now — the live
    /// ceiling [`Scheduler::admit_budget`] is clamped to on every read.
    ///
    /// Free reservation bytes plus reversibly-evictable KV, minus the hot KV the
    /// drain is skipping because it is pinned. The pinned discount is what keeps
    /// the forecast from reading its most optimistic exactly when the hot→warm
    /// drain has stalled: those bytes are counted as evictable but cannot be
    /// reclaimed at any price.
    ///
    /// The first term used to be a contest between three driver-derived
    /// estimates — governor headroom, the pool's reserved-but-free gap, and the
    /// allocator's own `init_free − pool_used − reserve` — clamped to whichever
    /// looked smallest, because each was wrong in a different regime. The worst
    /// was the reuse gap: admission once read 3045 MiB of it while `vram_free`
    /// was 0 and the pool held 15168 of 16375 MiB, admitted six prefills onto
    /// memory WDDM had already spilled, and the run aborted at ~3 tok/s. None of
    /// that survives the reservation. KV comes from regions that were claimed at
    /// startup, so what admission can spend is a count of the free ones, and no
    /// driver reading enters into it.
    ///
    /// Two corrections went with those estimates. One added what registered
    /// relievers claimed they could reversibly free; the other subtracted hot KV
    /// the drain was skipping because it was pinned, which the first had counted
    /// and could not actually reclaim. Both existed because the base number
    /// described *the card*. A free-region count describes what this process has
    /// claimed and not yet spent, so pinned KV is excluded by construction — it
    /// holds live regions — and evictable KV shows up as free regions the moment
    /// the relief pass ahead of admission actually evicts it. Measured, not
    /// forecast, which is why nothing has to be added back or discounted.
    pub(in super::super) fn admit_budget_ceiling(&self) -> u64 {
        // No forward reserve is subtracted here: admission holds the tier back
        // itself, at the width it is choosing (`Scheduler::admit_headroom`). The
        // setpoint IS subtracted — those regions are the relief pass's working
        // room, not admission's to spend.
        let Some((free, setpoint)) = self.kv_region_state(VramPhase::Load) else {
            return 0;
        };
        (free.saturating_sub(setpoint) as u64).saturating_mul(region_bytes())
    }

    /// Free KV regions right now, and the setpoint for `phase` — the two
    /// numbers every pressure and admission decision is made from.
    ///
    /// "Free" includes regions a standing transient tier has blocked
    /// (`stats.blocked`): every decision made from this pair concerns work
    /// scheduled for a *later* forward, and that forward's phase 0 releases the
    /// tier before any of its claims run. Counting only the tier-capped free
    /// count made every wave's own scratch read as KV pressure from the
    /// scheduler's seat, shedding sequences to relieve ground that was never
    /// occupied.
    ///
    /// `None` before the reservation exists, which the callers read as "no
    /// pressure, nothing to spend": there is no KV on the device yet to be
    /// under pressure about.
    pub(in super::super) fn kv_region_state(&self, phase: VramPhase) -> Option<(usize, usize)> {
        let stats = self.kv_regions()?;
        Some((
            stats.free + stats.blocked,
            setpoint_regions(phase, stats.total),
        ))
    }

    /// The KV side's region counters, or `None` before the reservation exists.
    pub(super) fn kv_regions(&self) -> Option<candle_nn::kv_cache::RegionStats> {
        let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() else {
            return None;
        };
        candle_nn::kv_cache::region_stats(gpu_id)
    }

    /// True when the KV side has fewer free regions than the setpoint — the
    /// signal to shed, and failing that to stop admitting.
    ///
    /// This used to be three gates in disjunction: a byte budget derived from
    /// `init_free − pool_used − reserve`, a driver-free floor qualified by how
    /// much the CUDA pool could still absorb by reuse, and a footprint gate on
    /// `pool_reserved` versus a compaction ceiling. Each existed because the
    /// other two were wrong in some regime, and the footprint gate needed a
    /// cooldown and a futility latch on top because a fragmented gap the engine
    /// kept reusing would otherwise report pressure on every scheduler loop.
    ///
    /// None of it survives the reservation. KV comes from regions claimed at
    /// startup, so the question "is there room?" has one exact answer that no
    /// driver reading enters into, and it cannot disagree with itself.
    ///
    /// Phase-independent default (`Load`, the wider setpoint). Prefer
    /// [`vram_under_pressure_for`](Self::vram_under_pressure_for) at call sites
    /// that know their phase.
    pub(in super::super) fn vram_under_pressure(&self) -> bool {
        self.vram_under_pressure_for(VramPhase::Load)
    }

    /// Phase-aware pressure signal — see [`VramPhase`] for why the setpoint
    /// differs by phase.
    pub(in super::super) fn vram_under_pressure_for(&self, phase: VramPhase) -> bool {
        self.kv_region_state(phase)
            .is_some_and(|(free, setpoint)| free < setpoint)
    }
}
