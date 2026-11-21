//! Takes a region for a new arena, sweeping and buying ground when short.
//!
//! A claim that finds no region returns `None`, which is a pressure signal
//! rather than an error: the caller evacuates a region or buys ground from the
//! weight side. Claims recycle the free list lowest-first and sweep empty
//! arenas before taking fresh ground, so long-lived arenas stay low and the
//! transient tier keeps its room. A claim that reaches the ceiling while a tier
//! stands is counted as a breach rather than refused, and every region comes
//! back zeroed. Exhaustion is recorded so the weight side refuses the next
//! negotiation instead of taking ground the KV side has just run out of.

use super::*;

/// Regions bought in one go when a claim runs the KV side out of ground.
///
/// The buy-side counterpart to [`kv_grow_step`], inverted for the same reason: a purchase
/// costs a device-wide quiesce (the weight side cannot hand over ground while a
/// kernel might still be reading it), so buying one region per claim would pay
/// that sync per arena. A section-quantize drain claimed eighteen regions in one
/// pass; at a region apiece that is eighteen full device syncs.
pub(super) const KV_BUY_STEP: usize = 8;

/// Take a region for an arena, or `None` when the KV side is full.
///
/// `None` is the pressure signal, not an error: the caller's response is to
/// evacuate a region (§3.8), which is the demotion the hot tier does anyway.
///
/// The region's bytes are zero on return — freshly-claimed ones from the
/// mapping touch, recycled ones from an explicit fill here, so a new tenant
/// never reads the last one's data (invariant 4).
pub(crate) fn claim_region(stream: &std::sync::Arc<CudaStream>) -> Result<Option<RegionHandle>> {
    // **Keep the frontier honest before taking virgin ground.** Claims recycle
    // the free list lowest-first, so the frontier only advances when the list
    // is empty — and if chunk-empty arenas exist at that moment, "empty" is a
    // bookkeeping artifact of churn, not real occupancy. Advancing anyway is
    // what pushes long-lived arenas into the top regions, and the transient
    // tier — anchored at the weight floor, measuring down — then loses that
    // ground for as long as those arenas live (measured on the Llama-2 MHA
    // gate: a span-end arena claimed at a churn-inflated 566/566 peak blocked
    // a 4-region tier with only 205 regions genuinely live). Sweep first, so
    // the frontier tracks the true live watermark.
    let free_list_empty = with_pool(stream, |pool| Ok(pool.free.peek().is_none()))?;
    if free_list_empty {
        super::super::backing::global_release_empty_arenas();
    }
    match with_pool(stream, |pool| try_claim(pool, stream))? {
        Claim::Got(handle) => return Ok(Some(handle)),
        // A tier is standing: the ground above it exists but belongs to a
        // running wave, and no amount of weight-side concession reaches it. The
        // answer is a narrower wave, not more ground.
        Claim::TierBlocked => return Ok(None),
        Claim::Exhausted => {}
    }
    // **Outside a wave, running out is a price, not a wall.** The span is one
    // reservation with a moving boundary, the cold tier holds a valid copy of
    // every expert, and so the KV side may buy the ground it needs by evicting
    // expert residency — which costs a reload and nothing else. The weight
    // zone's own floor is what finally refuses (`WeightZone::new`), and it is the
    // only refusal left in this path.
    if buy_ground(stream, KV_BUY_STEP)? == 0 {
        return Ok(None);
    }
    match with_pool(stream, |pool| try_claim(pool, stream))? {
        Claim::Got(handle) => Ok(Some(handle)),
        _ => Ok(None),
    }
}

/// What one attempt at [`claim_region`] found.
pub(super) enum Claim {
    Got(RegionHandle),
    /// Refused by a placed tier's base — unreachable until the wave ends.
    TierBlocked,
    /// No region below the boundary, and no tier to blame.
    Exhausted,
}

pub(super) fn try_claim(
    pool: &mut RegionPool,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<Claim> {
    {
        // **The ceiling binds the free list too.** A region freed while a tier
        // stands keeps its index, and that tier is placed below it. Handing it
        // out would put KV writes inside a running wave's intermediates.
        //
        // Between forwards there is no tier and the ceiling is the boundary
        // itself. It used to be the boundary less `tier_reserve` — the last
        // tier's width, held back so the next placement would find its ground
        // empty — and that reserve is gone. It could not do the job: the width
        // it withheld was one sample of a quantity that ranged from 4 to 34
        // regions across a single run, so it neither protected a wider tier nor
        // released the ground a narrower one left behind. What it did do was
        // refuse claims against unowned memory, and then the refusal escalated
        // into a boundary renegotiation that handed over the same ground by the
        // most expensive route available. [`place_transient`] buys what it needs
        // instead, on the same terms as any other claim.
        //
        // One comparison suffices: the heap is ordered lowest-index-first, so if
        // the lowest free region is above the ceiling then every free region is.
        let ceiling = pool.region_ceiling();
        let lowest_free_usable = pool.free.peek().is_some_and(|Reverse(i)| *i < ceiling);
        let (index, recycled) = if lowest_free_usable {
            let Reverse(idx) = pool.free.pop().expect("peeked");
            (idx, true)
        } else if pool.next < ceiling {
            // **A region claimed while a tier stands is an invariant breach.**
            //
            // Every region claim creates an arena, and an arena may only be
            // created between forwards — `BackingInner::arena_window` blocks
            // until then. So reaching here with a tier placed means some path
            // creates arenas without that gate, and the tier the running wave is
            // using was placed against a frontier that has just moved under it.
            //
            // Counted, loudly, rather than refused: refusing would turn a
            // bookkeeping hole into a failed forward, and the claim itself is
            // still served from ground nobody holds. The counter is a tripwire
            // that must read zero, not a tuning input — it used to be the latter,
            // and `fresh_per_forward_peak` grew a gap above the frontier to
            // accommodate exactly the claims this now forbids.
            if pool.transient_base.is_some() {
                pool.fresh_claims_during_wave += 1;
                log::warn!(
                    "reservation: fresh region claimed with the wave tier placed at {:#x} \
                     — an arena was created inside a wave (count {}). Every creation must \
                     pass BackingInner::arena_window, which waits for the gap between \
                     forwards.",
                    pool.transient_base.unwrap_or(0),
                    pool.fresh_claims_during_wave,
                );
            }
            pool.next += 1;
            (pool.next - 1, false)
        } else if pool.transient_base.is_some() {
            // **Blocked by the tier, which the boundary cannot fix.** The
            // ceiling is the tier's base while one is placed, and moving the
            // weight floor does not move a tier that is already standing — so
            // this shortfall is not demand the weight side can satisfy, and
            // asking it to pay is asking it to pay for nothing.
            //
            // It did exactly that. A section prefill wedged, the scheduler's
            // relief asked for ground on every retry, and each tier-blocked
            // refusal had counted as a region of boundary demand — so the weight
            // side conceded, one region at a time, evicting six experts a step,
            // until the expert zone was **empty**: `slots=0 weights=0MiB`, every
            // expert gone, and the ceiling still exactly where it started
            // because `transient_base` had never moved.
            //
            // What this shortfall calls for is a narrower wave or a released
            // tier, and `refusals_during_wave` is what says so.
            //
            // **Like the branch above, this now reads zero or something is
            // wrong.** A claim reaching the ceiling with a tier standing is a
            // claim that arrived inside a wave, which `arena_window` forbids.
            pool.refusals_during_wave += 1;
            if super::super::alloc::arena_stats_enabled() {
                eprintln!(
                    "[tier-refusal] region claim refused with the tier placed \
                     at {:#x} (ceiling {ceiling}, next {}, total {}) — count {}",
                    pool.transient_base.unwrap_or(0),
                    pool.next,
                    pool.total,
                    pool.refusals_during_wave,
                );
            }
            return Ok(Claim::TierBlocked);
        } else {
            // Exhausted with no tier standing. The caller buys ground and comes
            // back; this attempt only reports what it found.
            //
            // **The weight side must hear about it.** Exhaustion is the KV side
            // asking for more and being told no, and
            // [`RegionPool::spare_regions`] reads this flag to refuse the next
            // negotiation whatever its occupancy says. Setting it only on a
            // *completed* purchase would leave the guard dead in the one case
            // that matters most — the KV side refused and the weight side
            // unwilling to sell — and let the weight side take ground from a KV
            // side that had just been turned down.
            pool.growth.note_demand();
            return Ok(Claim::Exhausted);
        };
        issue_region(pool, stream, index, recycled).map(Claim::Got)
    }
}

/// The lowest `k` consecutive region indices a run may take below `ceiling`:
/// each either on the free list or at or past `next`, which is fresh ground.
/// `None` when no such run fits.
///
/// Pure index arithmetic over the free set, so the rule — lowest run first, fresh
/// ground counted as free, nothing at or above the ceiling — is tested without a
/// device.
pub(super) fn lowest_run(
    free: &std::collections::HashSet<usize>,
    next: usize,
    ceiling: usize,
    k: usize,
) -> Option<usize> {
    if k == 0 {
        return None;
    }
    let available = |i: usize| i >= next || free.contains(&i);
    let mut start = 0;
    while start + k <= ceiling {
        match (start..start + k).find(|&i| !available(i)) {
            None => return Some(start),
            Some(blocked) => start = blocked + 1,
        }
    }
    None
}

/// Whether `k` consecutive regions ending at or below region `below` are free now
/// — [`SpanClaims::claim_run_below`]'s question, asked without a claims window,
/// which would hand back a standing tier to learn the answer is no.
pub(crate) fn run_free_below(stream: &std::sync::Arc<CudaStream>, k: usize, below: usize) -> bool {
    with_pool(stream, |pool| {
        let free: std::collections::HashSet<usize> =
            pool.free.iter().map(|Reverse(i)| *i).collect();
        Ok(lowest_run(&free, pool.next, pool.region_ceiling().min(below), k).is_some())
    })
    .unwrap_or(false)
}

/// [`try_claim`] for `k` consecutive regions — a tenant whose buffer is wider than
/// one region — ending at or below region `below`. The run is taken whole or not at
/// all: a region of it that cannot be issued puts back every one already issued.
///
/// A refusal is KV demand only when the pool's own ceiling bound it; a tighter
/// `below` is a tenant asking to move down, which wants no new ground.
pub(super) fn try_claim_run(
    pool: &mut RegionPool,
    stream: &std::sync::Arc<CudaStream>,
    k: usize,
    below: usize,
) -> Result<Option<Vec<RegionHandle>>> {
    let ceiling = pool.region_ceiling();
    let free: std::collections::HashSet<usize> = pool.free.iter().map(|Reverse(i)| *i).collect();
    let Some(start) = lowest_run(&free, pool.next, ceiling.min(below), k) else {
        if pool.transient_base.is_none() && below >= ceiling {
            pool.growth.note_demand();
        }
        return Ok(None);
    };
    let run = start..start + k;
    // Off the free list in one rebuild; fresh indices past `next` move it.
    pool.free.retain(|Reverse(i)| !run.contains(i));
    if run.end > pool.next {
        pool.next = run.end;
    }
    let mut issued = Vec::with_capacity(k);
    for index in run.clone() {
        match issue_region(pool, stream, index, free.contains(&index)) {
            Ok(handle) => issued.push(handle),
            Err(e) => {
                // The issued handles' own drops would take the pool lock, which
                // is ours: they are put back by hand, exactly as a drop would.
                for h in issued {
                    pool.owned[h.index] = false;
                    pool.dirt[h.index] = Dirt::any(pool.quiesce_epoch);
                    pool.free.push(Reverse(h.index));
                    pool.live -= 1;
                    std::mem::forget(h);
                }
                // `issue_region` puts back the region it refused; the ones after
                // it were taken off the list (or past `next`) and never issued.
                for later in index + 1..run.end {
                    pool.free.push(Reverse(later));
                }
                return Err(e);
            }
        }
    }
    Ok(Some(issued))
}
