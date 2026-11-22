//! Prices and reserves the transient tier that each forward runs in.
//!
//! A forward's phases carve from one tier, placed once for the forward at the
//! width its plan prices. The tier's ground is held from the forward's first
//! phase until the next forward places its own, and is not released at each
//! phase boundary: releasing it there would let the region ceiling move
//! mid-sweep and put one layer's phases and the next layer's at different
//! offsets. A walk that runs between forwards raises the recorded plan only as
//! far as the standing tier holds, so the room the forward reserved is kept
//! without buying more.

use super::*;

/// Price **and reserve** the transient tier for the forward about to run.
///
/// Called once per forward, after its KV is admitted and before it opens its
/// first phase. The three figures come from `WavePlan::phase_bytes` at the
/// admitted row count, so the tier is sized to **this** wave rather than to the
/// widest one the engine can run — a twenty-session decode needs a few megabytes
/// where the fixed constants reserved 912 MiB, and the difference is what the
/// weight side gets to hold.
///
/// # The reservation is per forward, and that is the whole point
///
/// This used to only record a size, with [`begin_wave`] deciding the address on
/// each phase and [`release_if_last`] giving the ground back when that phase's
/// last guard dropped. A forward opens and closes its phases many times, so the
/// tier's ground was released and re-taken between every one of them — and in
/// each of those gaps the region ceiling lifted, another thread could claim a
/// fresh region, and the next phase's tier landed one region further along.
///
/// A tier that walks forward mid-sweep puts layer *N*'s FFN extent and layer
/// *N+1*'s attention extent at **different** offsets, which is not the situation
/// [`Reclaim::StreamOrdered`] argues about: that argument is a phase reusing
/// *its own* span with a whole other phase's same-stream work in between. Two
/// different extents, on two streams, have nothing ordering them. It cost
/// `Q8_0 x20` every session's output while the region pool looked untouched.
///
/// So the address is decided **here**, once, and held for the forward: the
/// ceiling stands across the inter-phase gaps, `begin_wave` rebases onto an
/// address it does not choose, and `release_if_last` returns the arenas to
/// nothing without returning the ground.
///
/// A forward that begins while a guard from the previous one is still live keeps
/// that reservation rather than moving it — the tier cannot move under a live
/// generation, and refusing here would fail a forward for a bookkeeping reason.
/// Hand back the previous forward's tier **before this one claims its KV.**
///
/// The tier's ground is deliberately held past the guard's drop
/// ([`release_if_last`]) so a region freed between forwards cannot be handed to
/// someone else and move the next tier. [`plan_wave_transient`] is what returns
/// it — and that is one phase too late.
///
/// A forward runs admit *first*: §7 phase 1 claims every KV slot the wave will
/// write so the frontier is final when phase 2 prices the tier against it. But
/// the previous tier is still standing during phase 1, and
/// [`super::region_pool::claim_region`] caps the pool at its base — so admit is
/// refused by a reservation belonging to a forward that has already finished.
///
/// While every wave succeeds this is invisible: the tier is re-placed each time
/// at the same width, and admit's claims fit below it. It becomes fatal the
/// moment one wave fails. The failed wave's tier stays, the retry's admit is
/// refused by it, that attempt fails too, and the engine spins — which is
/// exactly what the daemon did, at 1,500 refusals a second, with `class 4096 B`
/// named in the error because active K is what admit claims.
///
/// So the release moves here, ahead of admit. The guard is the same one
/// `plan_wave_transient` applies: a forward that begins while a generation from
/// the previous one is still live keeps that reservation rather than moving it,
/// because the tier cannot move under a live generation.
///
/// # Every caller outside a forward, not just the next one
///
/// Admit is the *first* thing a stale tier blocks, not the only one. The
/// scheduler's relief ladder runs in the same inter-forward gap and claims
/// regions on nearly every rung, and one consequence there is worse than a
/// refused claim: the pool's region ceiling reads `transient_base` while a tier
/// is placed, and that is an *address*, fixed where the last forward put it.
/// Move the weight boundary and it does not follow — so **a placed tier makes
/// the ceiling deaf to the boundary**, and the ladder's last
/// rung, the only one that adds ground rather than recycling it, hands over
/// weight-side ground that no rung above it can reach.
///
/// The contract is therefore "call this before touching the region pool from
/// outside a forward", and the `live_generations` guard is what makes it safe to
/// state that broadly: a caller that is wrong about being outside a forward
/// finds a live generation and changes nothing.
pub fn end_wave_transient(stream: &Arc<CudaStream>) {
    let stream = stream.clone();
    let _ = with_wave_domain(&stream, |domain| {
        if domain.live_generations > 0 {
            return Ok(());
        }
        if domain.placed_at.take().is_some() {
            domain.placed_bytes = None;
            release_transient(&stream);
            domain.reserved_by_forward = false;
        }
        Ok(())
    });
}

/// Price and reserve this forward's transient tier.
///
/// # Three lock scopes, and the middle one is deliberate
///
/// The release and the record happen under the wave-domain lock; the
/// **placement does not**, and it must not. [`place_transient`] buys ground from
/// the weight side when live arenas hold the tier's footprint, that purchase
/// blocks on the expert pipeline thread, and the boundary move it performs calls
/// [`wave_is_live`] — which takes this very lock. Holding it across the purchase
/// deadlocks the two threads against each other.
///
/// So there is a window between releasing the old tier and placing the new one
/// in which another thread may claim a *slot* in a region an arena already
/// holds. That is not a hazard: the placement measures the arena frontier, and
/// filling an existing arena does not move it. What may not happen in that
/// window is a **new arena**, which does move the frontier — hence the wait on
/// [`enter_arena_window`] below, taken before the frontier is read and held
/// until the tier is recorded.
pub fn plan_wave_transient(stream: &Arc<CudaStream>, per_phase: [usize; 3]) -> Result<()> {
    let (per_phase, scratch) = with_launch_scratch(stream, per_phase);
    place_plan(stream, per_phase, scratch)
}

/// [`plan_wave_transient`] for a plan that already carries `scratch` bytes of
/// launch-scratch reserve in its forward-wide span.
pub(super) fn place_plan(
    stream: &Arc<CudaStream>,
    per_phase: [usize; 3],
    scratch: usize,
) -> Result<()> {
    let stream = stream.clone();
    // The previous forward's ground goes back before this one's frontier is
    // measured, so the reservation is taken against a partition that already
    // reflects everything admit claimed.
    let place = {
        // **Anyone mid-way through creating an arena finishes first.** The
        // placement below reads the arena frontier; a half-created arena would
        // move it after the read.
        let mut map = await_arena_windows(lock_domains(), &stream);
        let (_, domain) = domain_entry(&mut map, &stream);
        domain.planned = Some(per_phase);
        domain.scratch_priced = scratch;
        domain.plan_pending = true;
        if domain.live_generations > 0 {
            false
        } else {
            if domain.placed_at.take().is_some() {
                domain.placed_bytes = None;
                release_transient(&stream);
            }
            true
        }
    };
    wave_gate().notify_all();
    if !place {
        return Ok(());
    }
    // Outside the lock. A failure here leaves no tier standing and
    // `reserved_by_forward` clear, so the forward that fails takes nothing with
    // it — the next one starts against the whole partition.
    let bought = tier_footprint(per_phase);
    let base = place_transient(&stream, bought)?;
    with_wave_domain(&stream, |domain| {
        domain.placed_at = Some(base);
        domain.placed_bytes = Some(bought);
        domain.reserved_by_forward = true;
        Ok(())
    })
}

/// `per_phase` with the forward-wide span widened by the launchers' scratch
/// reserve on `stream`'s device — what recent forwards' launchers asked for
/// (`candle::cuda_backend::scratch_reserve`) — and normalised; with the reserve.
/// A plan prices the forward's own buffers; the launchers size theirs inside the
/// launch, so the span carries what they have been seen to need, and they carve
/// no more than that ([`carve_forward_scratch`]).
///
/// Normalised once, here, so the recorded plan, the tier purchase, and the span
/// layout in `begin_wave` all see the same rounded figures — a raw sum against a
/// rounded layout would place a tier the last span overruns.
pub(super) fn with_launch_scratch(
    stream: &Arc<CudaStream>,
    per_phase: [usize; 3],
) -> ([usize; 3], usize) {
    let scratch = scratch_reserve(stream.context().ordinal());
    (launch_plan(per_phase, scratch), scratch)
}

/// The plan that covers both `standing` and `wanted`, phase by phase.
pub(super) fn covering_plan(standing: [usize; 3], wanted: [usize; 3]) -> [usize; 3] {
    [
        standing[0].max(wanted[0]),
        standing[1].max(wanted[1]),
        standing[2].max(wanted[2]),
    ]
}

/// Raise the recorded plan so a caller that runs **between forwards** has room.
///
/// A forward prices its tier from its own width, and the tier then stands for
/// whoever runs before the next forward. A draft walk is such a caller, and its
/// cohort is not the forward's: it is every sequence that came out of accept
/// drafting together, which can be wider than the forward that preceded it. The
/// head's attention step then carves `rows × (per-row chain)` from a span priced
/// for fewer rows and exhausts it — measured on the 35B-A3B, a 2-row cohort
/// asking 9,216 B at offset 61,952 of a 64,256 B span, which is one row's price.
///
/// While a tier stands, the recorded plan is raised to the per-phase maximum of
/// itself and `per_phase` **if that still fits the bytes the tier holds**: the
/// spans are re-laid inside the standing tier and nothing is bought, so the room
/// the forward reserved in the phases the walk does not outgrow is kept. A plan
/// that already covers `per_phase` changes nothing and moves nothing.
///
/// **A merge the standing tier cannot hold places the walk's own width.**
/// Placing releases the standing tier first, so a merged plan keeps none of the
/// forward's room — it only re-buys that forward's whole footprint, against a
/// partition whose KV has grown since. Measured in zend: a draft walk after an
/// 8K-row prefill outgrew one phase by 86 KB, merged to 8,992,587,776 B, and was
/// refused by a gap of 8,992,501,760 B; six files failed their ingest on it.
/// Nothing after a walk needs the previous forward's room: the next forward
/// prices its own tier, and a rewind's replay runs on the tier of the forward it
/// rewinds.
///
/// **With no tier standing, the caller prices for itself alone.** The ground
/// went back to the weight side, and with it whatever the last forward reserved
/// — there is no room left to preserve. Merging anyway re-bought the last
/// forward's whole tier for a walk a fraction of its width: measured on
/// Flash-Next's C5×8 gate, a draft walk after the prefill asked for the
/// prefill's 4.0 GB, against a partition whose KV had grown since and whose
/// weight side would not concede mid-flight, and the wave failed.
///
/// **Hold the returned [`Covering`] for the whole walk.** While it lives, an
/// arena the walk creates hands the tier back but leaves the walk's width as
/// the plan ([`WaveDomain::covering`]); dropped, the width goes with it, so no
/// guard after the walk lays its spans out at a walk's width.
pub fn cover_wave_transient(stream: &Arc<CudaStream>, per_phase: [usize; 3]) -> Result<Covering> {
    let (wanted, scratch) = with_launch_scratch(stream, per_phase);
    let (ordinal, target) = {
        let mut map = lock_domains();
        let (ordinal, domain) = domain_entry(&mut map, stream);
        domain.covering = Some(wanted);
        domain.covering_scratch = scratch;
        let target = cover_target(
            domain.planned,
            domain.placed_at.and(domain.placed_bytes),
            wanted,
        );
        // Re-laid in place: `begin_wave` keeps a standing tier that holds the
        // recorded plan and lays the spans out from it.
        if let Cover::Relay(plan) = target {
            domain.planned = Some(plan);
        }
        // **A kept or re-laid span carves the lesser of the two reserves.** Its
        // forward-wide span holds both the standing plan's buffers and their
        // reserve and the walk's and theirs, but not either one's buffers beside
        // the other's reserve: a walk whose own buffers are wider than the
        // standing forward's, carving that forward's larger reserve, would reach
        // past what the span holds.
        if !matches!(target, Cover::Place(_)) {
            domain.scratch_priced = domain.scratch_priced.min(scratch);
        }
        (ordinal, target)
    };
    // Built first, so a refused placement clears the width on the way out.
    let covering = Covering { ordinal };
    match target {
        Cover::Place(plan) => place_plan(stream, plan, scratch)?,
        Cover::Keep | Cover::Relay(_) => {}
    }
    Ok(covering)
}

/// A walk's claim on its own width; see [`cover_wave_transient`].
#[must_use = "the walk's width lasts only while this is held"]
pub struct Covering {
    pub(super) ordinal: usize,
}

impl Drop for Covering {
    fn drop(&mut self) {
        let mut map = lock_domains();
        if let Some(domain) = map.get_mut(&self.ordinal) {
            domain.covering = None;
        }
    }
}

/// What [`cover_wave_transient`] does for a walk — see there for the rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Cover {
    /// The standing tier and its plan already hold the walk.
    Keep,
    /// Record this plan; it fits inside the standing tier, so nothing moves.
    Relay([usize; 3]),
    /// Place a tier for this plan.
    Place([usize; 3]),
}

/// The [`Cover`] action for `wanted`, given the recorded plan and the bytes the
/// standing tier holds (`None` when no tier stands).
pub(super) fn cover_target(
    planned: Option<[usize; 3]>,
    standing_bytes: Option<usize>,
    wanted: [usize; 3],
) -> Cover {
    match (planned, standing_bytes) {
        (Some(standing), Some(bytes)) => {
            let merged = covering_plan(standing, wanted);
            if merged == standing {
                Cover::Keep
            } else if tier_footprint(merged) <= bytes {
                Cover::Relay(merged)
            } else {
                Cover::Place(wanted)
            }
        }
        _ => Cover::Place(wanted),
    }
}

/// What `stream`'s wave domain stands at now: the plan recorded for it, each
/// phase arena's laid-out capacity, and the generations live — read for a
/// refusal or an overrun to say whether the spans were laid out for the plan
/// that ran on them.
pub fn wave_domain_layout(stream: &Arc<CudaStream>) -> Option<([usize; 3], [usize; 3], usize)> {
    let mut map = lock_domains();
    let (_, domain) = domain_entry(&mut map, stream);
    let planned = domain.planned?;
    let mut laid = [0usize; 3];
    for (i, arena) in domain.arenas.iter().enumerate().take(3) {
        laid[i] = arena.stats().2;
    }
    Some((planned, laid, domain.live_generations))
}
