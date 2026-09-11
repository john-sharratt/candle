//! Whether the weight side may take KV ground, and how much.
//!
//! The decision only — the measurements it reads are gathered by
//! [`super::region_pool`], which owns the reservation, the free list and the
//! transient tier. Splitting the two is the same move [`super::weight_zone`]
//! makes for the mirror side, and for the same reason its header gives: *keeping
//! the policy out means the whole module tests without a GPU, a model, or a
//! routing trace*.
//!
//! # Why this is worth its own file
//!
//! The partition's defects have all been **trajectory** defects. No single call
//! returned a wrong number; a sequence of individually defensible answers walked
//! the boundary somewhere bad and left it there — the ratchet held 34 of 64
//! layers streaming through two configs that needed a quarter of the KV, and
//! every unit test passed throughout.
//!
//! A trajectory is only testable if it can be *run*, and running one against the
//! pool means a device, a process-global lock, and a few hundred milliseconds per
//! scenario. Against this it is arithmetic: a soak test of thousands of forwards
//! across every card size and model shape costs milliseconds and needs no GPU.
//! `docs/vram_partition_behavioural_tests.md` is the catalogue that buys.
//!
//! # The shape of the answer
//!
//! Three guards, and every one is a statement about **now** rather than a
//! forecast:
//!
//! 1. **Observation.** Nothing is spare until something has been demanded.
//! 2. **The derivative.** Demand rising, or a purchase since the last look, means
//!    whatever is free is about to be taken.
//! 3. **Occupancy.** What the KV side does not hold, less a slack margin.
//!
//! There used to be a fourth — a windowed maximum of past demand — and removing
//! it is what un-stuck the boundary. See [`GrowthPolicy::spare`].

/// Most regions the weight side may take in one negotiation.
///
/// **Half of what the guards found spare, never fewer than `min_grant`.**
///
/// This was a flat eight, on the reasoning that "growth is a step, not a jump,
/// because each region it takes may have to be given back — and giving back
/// costs an eviction or a relocation, while not taking costs only the residency
/// it would have bought for one more pass". Both halves of that turned out to be
/// measurably wrong on the 3.6-35B gate:
///
/// - **Giving back is free in practice.** Instrumented over a full gate: twelve
///   KV purchases, *zero* refused. Every time the KV side wanted ground back it
///   got it, and the give-back path is a reload, not a loss.
/// - **Not taking costs the whole workload, not one pass.** [`GrowthPolicy::spare`]
///   found ~143 regions genuinely spare on each of the twenty-one negotiations
///   that got past the guards, and handed over eight. At that rate the boundary
///   converges long after the run it was supposed to help has finished.
///
/// So the step is geometric rather than fixed. Halving keeps the hedge — a
/// negotiation never takes everything it is offered — while letting it shrink as
/// evidence accumulates: each pass that takes ground without the KV side buying
/// it back is evidence the last one was safe.
///
/// The safety net underneath is admission, not this constant: the scheduler's
/// ceiling is read live from free regions, so ground given to the weights
/// narrows what admission accepts rather than failing anything.
///
/// # The floor is the caller's allocation unit, not a constant
///
/// `min_grant` used to be a hard `8`, which is the **expert cache's** unit — an
/// expert slot — written into a function two different consumers share. A
/// consumer whose unit is larger can be handed a grant it cannot spend, and the
/// geometric convergence above quietly stops working: "each pass that takes
/// ground without the KV side buying it back is evidence the last one was safe"
/// assumes each pass *takes* the ground. A consumer that discards its grant
/// accumulates no evidence, so the next pass is offered the same unusable number
/// forever.
///
/// Measured on the 27B, whose unit is a ~154 MiB layer — about ten regions
/// against this floor of eight: over one gate run the pool granted 396 regions
/// and the layer zone applied 198 of them, discarding **3.1 GiB** of offered
/// ground in grants too small to buy a single layer. The `.min(spare)` clamp
/// keeps the raised floor honest — a caller is never handed more than is spare,
/// only all of it when all of it is barely enough.
///
/// # Halving is load-bearing, and taking the whole offer was measured worse
///
/// The layer zone's cost of *undershooting* is a ~160 MiB synchronous transfer
/// per missing layer per forward, which reads like an argument for taking
/// everything spare in one negotiation. It is not: tried on the 27B, taking the
/// full offer overshot, the next admission bought the ground straight back, and
/// the purchase set the pressure guard that refuses the *next* negotiation.
/// Applied grants fell from four to two and the zone settled
/// a layer lower — the churn cost more than the slower convergence it was meant
/// to avoid. The hedge is what stops that loop, and both consumers want it.
/// # A grant below the floor is refused, not clamped
///
/// The clamp used to be `.min(spare)`, which returns a grant *below*
/// `min_grant` whenever `spare < min_grant` — precisely the case the floor was
/// added to close. On the 27B (`min_grant = 26`, a 202 MiB layer over 16 MiB
/// regions) a spare of 20 was handed over, bought one layer, and was then
/// discarded by the zone's two-layer hysteresis with nothing applied — while
/// [`GrowthPolicy::spare`] had already consumed the negotiation. The zone did
/// not grow and the pool accumulated no evidence, for every forward where spare
/// sat in `1..min_grant`.
///
/// So an offer that cannot buy the caller's unit is **zero**. That is the honest
/// answer: the caller would discard it, and reporting it as a grant makes a
/// refusal look like a take in every counter that reads this.
pub fn kv_grow_step(spare: usize, min_grant: usize) -> usize {
    if spare < min_grant {
        return 0;
    }
    (spare / 2).max(min_grant).min(spare)
}

/// Regions the KV side must buy from the weight side so that, after its
/// `claims` regions of K/V and recurrent store are taken, `tier` regions still
/// stand contiguous at the arena frontier — given the KV side holds `free`
/// regions anywhere and `gap` of those lie between the frontier and the weight
/// floor.
///
/// The mirror of [`kv_grow_step`]: that one answers what the weight side may
/// take, this one what it must give back.
///
/// Claims recycle the free list lowest-first, so they spend the regions
/// scattered below the frontier before they reach into the gap; the tier stands
/// only in the gap, so what the claims eat of it has to be bought back. Buying
/// moves the floor right, which adds to the gap and the free list at once.
///
/// The first shape of this — the larger of `claims + tier − free` and
/// `tier − gap` — let an admission's own claims consume the gap it had just
/// checked: run 5 admitted 42 sections, each measured the gap as sufficient,
/// each then claimed its store from the top of the span, and the wave's tier
/// found the gap three regions short with twelve regions free below it. No
/// forward ran for the rest of the run.
///
/// **Both buyers reach this.** The scheduler's admission is one
/// (`Scheduler::buy_kv_ground`); a driver that has no admission stage is the
/// other, and the batched forward gate is exactly that. A harness that creates
/// twenty recurrent stores without buying their ground finds the span full with
/// the layer zone whole and 10.5 GiB it would have conceded on contact — which
/// reads identically to a span that is genuinely out of room.
///
/// Pure, so the arithmetic is tested without a device.
pub fn kv_ground_shortfall(claims: usize, tier: usize, free: usize, gap: usize) -> usize {
    let scattered = free.saturating_sub(gap);
    let gap_eaten = claims.saturating_sub(scattered);
    claims
        .saturating_sub(free)
        .max(tier.saturating_add(gap_eaten).saturating_sub(gap))
}

/// What the pool measures for one negotiation.
///
/// Gathered at phase 0, where the present is knowable exactly: the tier has been
/// released, no wave generation is open, and empty arenas have just been swept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Occupancy {
    /// Regions held by an arena, empty or not.
    pub live: usize,
    /// Regions on the free list below the tier ceiling — claimable right now.
    pub free_below_ceiling: usize,
    /// Regions the tier's ceiling puts out of reach, plus what the weight side
    /// already holds. Idle rather than occupied: the boundary moves only with no
    /// wave open, so a standing tier's bytes are dead until phase 0 releases
    /// them.
    pub ceiling_blocked: usize,
    /// The transient tier's current footprint, bytes.
    pub tier_bytes: usize,
    /// The widest tier this process has stood, bytes.
    ///
    /// The high-water rather than the live figure, because the live one is zero
    /// every time a negotiation runs: every caller reaches it on the line *after*
    /// `end_wave_transient`. A demand that never contains a tier would let the
    /// weight side take exactly the ground the next wave needs — ground it
    /// cannot hand back mid-forward, because the floor is refused while a wave
    /// generation is open.
    pub tier_high_water: usize,
    /// Bytes of the tier the next wave is guaranteed: the least forward worth
    /// running, as admission prices it (`region_pool::set_least_tier_bytes`).
    ///
    /// **The term this policy was written without.** The spare below used to be
    /// offered whole, with the next wave's tier undeducted because the signature
    /// could not see it — documented as costing "churn rather than failure", the
    /// next admission buying back for its tier the ground that had just been
    /// given away.
    ///
    /// Measured, that churn is not a rounding error: 19,878 grows against 20,018
    /// concessions in one run, ~4.6 full evict-and-reload cycles a second, each
    /// trading ~19 regions and ~165 expert slots. Throughput fell from 12,569 to
    /// ~40 tok/s and 14 of 353 directories completed. The demand guard above
    /// cannot see it either: the loop returns demand to where it started, so it
    /// is flat rather than rising and `Refusal::Pressure` never fires.
    ///
    /// The least tier, not the widest recent one and not the high-water. Both
    /// of those preserve whatever gap the last wide wave packed itself into —
    /// the tier is sized to the gap, so deducting it is a ratchet the weight
    /// side never climbs back out of.
    pub tier_planned: usize,
}

/// Why a negotiation answered as it did.
///
/// The whole point of naming these: a zero that is "the mechanism is inert" and a
/// zero that is "the ground is genuinely spoken for" are the same number and
/// completely different findings. Three rounds of this session's debugging were
/// spent inferring which, from counters one layer away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// No workload has run yet, so there is nothing to measure.
    Observing,
    /// Demand is rising, or the KV side bought ground since the last look.
    Pressure,
    /// The KV side is holding it. This is the only refusal that means the
    /// partition is working and the answer is simply no.
    Occupied,
}

/// The growth direction's decision and the state it carries between calls.
#[derive(Debug, Clone, Default)]
pub struct GrowthPolicy {
    /// Whether any demand has ever been observed.
    seen_demand: bool,
    /// Demand at the previous negotiation, so this one can see which way it is
    /// moving — the one signal occupancy cannot give.
    last_demand: usize,
    /// Set when the KV side asks for more ground: a completed purchase, or a
    /// claim that found the pool exhausted. Cleared by the next negotiation.
    asked_since_negotiation: bool,
}

impl GrowthPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that the KV side asked for ground.
    ///
    /// Its own voice, and the next negotiation must hear it: a side that has just
    /// run out is not a side with ground to spare, whatever occupancy says a
    /// moment later.
    pub fn note_demand(&mut self) {
        self.asked_since_negotiation = true;
    }

    /// Regions the weight side may take: what the KV side is **not using now**,
    /// less `slack`, and nothing else.
    ///
    /// # Why the present, and not a forecast of it
    ///
    /// This answered against a sliding-window maximum of past KV demand for as
    /// long as the boundary existed, on the reasoning §7a of
    /// `docs/archived/elastic_vram_partition.md` states outright: *"fast to
    /// concede, slow to take: being short of KV **fails a forward**, being short
    /// of experts is a slowdown, and the two are not worth trading
    /// symmetrically."*
    ///
    /// **Being short of KV no longer fails a forward.** Admission prices every
    /// row it takes and asks the weight side for exactly the shortfall before
    /// the wave (`request_kv_ground`), and the weight side concedes on contact.
    /// Measured over a full 27B gate: thirty purchases, **zero refused**. The
    /// forecast was insurance against a loss that can no longer occur, and it
    /// was not free.
    ///
    /// What it cost was a **ratchet**. Shrink reads the present exactly —
    /// admission evicts weights on contact — while grow consulted a forecast, so
    /// a wide cohort drove the zone down in seconds and no idle time brought it
    /// back: the mark remembered a peak the workload had left behind while
    /// occupancy said the ground was free. Removing it took the 27B to full
    /// residency on every single-context config and 26 → 35 layers on the widest.
    ///
    /// The `slack` term stays, because it covers the one quantity genuinely in
    /// the future: persistence's quantize destinations, which are not claimed
    /// when the boundary moves. §13b of the same document records that trying to
    /// make *that* exact was refuted, for the good reason that it has not
    /// happened yet.
    pub fn spare(
        &mut self,
        occ: Occupancy,
        slack: usize,
        region_bytes: usize,
    ) -> Result<usize, Refusal> {
        let observing = !self.seen_demand;
        let tier = occ.tier_bytes.max(occ.tier_high_water);
        let demand = occ.live + tier.div_ceil(region_bytes.max(1));
        if demand > 0 {
            self.seen_demand = true;
        }
        if observing {
            return Err(Refusal::Observing);
        }
        let rising = demand > self.last_demand;
        let bought = std::mem::take(&mut self.asked_since_negotiation);
        self.last_demand = demand;
        if rising || bought {
            return Err(Refusal::Pressure);
        }
        // **`ceiling_blocked` is zero here, and this term is therefore
        // `free_below_ceiling` alone.**
        //
        // A negotiation is only reachable from `reclaim_spare_ground`, on the
        // line after `end_wave_transient` — so no tier stands, the region ceiling
        // is the pool's size, and nothing is above it. The field is kept in
        // [`Occupancy`] because it is the honest description of what the caller
        // measured, not because it can be non-zero at this call site.
        //
        // That matters because the comment here used to argue the opposite: that
        // the tier need not be deducted since `ceiling_blocked` *is* the tier's
        // ground. It is, during a wave — and never at the one moment this runs.
        // So the next wave's tier is genuinely unaccounted for, and the honest
        // statement is that this offers ground the next admission may then
        // have to buy back for its tier, at the cost of churn rather than
        // failure.
        //
        // Deducting `transient_high_water` was tried and is worse: it is the
        // widest tier the process ever stood (up to the full reservation), not
        // the next one's price, and on the 27B it cut applied grants from 17 to 4
        // and cost five layers of residency. The right term is this wave's
        // planned tier, which `WavePlan` knows and this signature does not — so
        // it is left undeducted deliberately, and named.
        // **The tier's own ground is not spare.** Offering it is what makes the
        // weight side take ground the very next forward must buy back, and the
        // buy-back is an evict-and-reload of the expert slots that stood on it.
        let by_occupancy = occ.free_below_ceiling + occ.ceiling_blocked;
        let tier_regions = occ.tier_planned.div_ceil(region_bytes.max(1));
        match by_occupancy
            .saturating_sub(slack)
            .saturating_sub(tier_regions)
        {
            0 => Err(Refusal::Occupied),
            n => Ok(n),
        }
    }
}

#[cfg(test)]
mod ground_shortfall_tests {
    use super::kv_ground_shortfall;

    /// Enough free ground everywhere it is needed: nothing is bought. The
    /// claims fit in the regions scattered below the frontier, so the gap is
    /// untouched and already holds the tier.
    #[test]
    fn nothing_is_bought_when_the_claims_fit_below_the_gap_and_the_gap_holds_the_tier() {
        assert_eq!(kv_ground_shortfall(10, 4, 20, 6), 0);
        assert_eq!(
            kv_ground_shortfall(0, 0, 0, 0),
            0,
            "a free admission buys nothing"
        );
    }

    /// Claims past the whole free list: the difference is bought, and with no
    /// tier to stand that is all.
    #[test]
    fn claims_past_the_free_list_buy_the_difference() {
        assert_eq!(kv_ground_shortfall(10, 0, 7, 0), 3);
    }

    /// **The tier needs the gap, not the free list.** Plenty of free regions
    /// scattered below the frontier do not place a tier; the gap decides.
    #[test]
    fn a_tier_wider_than_the_gap_is_bought_even_with_free_regions_elsewhere() {
        assert_eq!(kv_ground_shortfall(4, 4, 40, 1), 3);
    }

    /// **Claims that reach into the gap are bought back for the tier.** Run 5:
    /// each of 42 admissions saw a gap that held its tier, then claimed its
    /// store from the top of the span and left the next wave's tier three
    /// regions short with twelve regions free below it.
    #[test]
    fn claims_that_would_eat_the_gap_are_bought_back() {
        // 5 free, 2 of them the gap: 10 claims spend the 3 scattered, then eat
        // the gap, then need 5 more — and the tier of 6 must still stand after.
        assert_eq!(kv_ground_shortfall(10, 6, 5, 2), 11);
        // 3 claims fit in the 3 scattered: only the tier's own shortfall.
        assert_eq!(kv_ground_shortfall(3, 6, 5, 2), 4);
        // 4 claims take the 3 scattered and one of the gap's 2: the tier of 2
        // needs that one back.
        assert_eq!(kv_ground_shortfall(4, 2, 5, 2), 1);
    }

    /// **The batched forward gate's case**, which has no admission stage: the
    /// KV side holds a 32-region zone whole, twenty recurrent stores want 158
    /// regions, and a 550 MiB tier must still stand. Nothing here is
    /// satisfiable from the free list, so the whole demand is a purchase — and
    /// a driver that never makes it reads the refusal as a full span.
    #[test]
    fn a_driver_with_no_admission_buys_its_whole_demand() {
        assert_eq!(kv_ground_shortfall(158, 35, 32, 32), 161);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: usize = 16 * 1024 * 1024;

    fn steady(live: usize, free: usize) -> Occupancy {
        Occupancy {
            live,
            free_below_ceiling: free,
            ceiling_blocked: 0,
            tier_bytes: 0,
            tier_high_water: 0,
            tier_planned: 0,
        }
    }

    /// The first negotiation never grants, whatever the card looks like — a span
    /// nothing has run on is not a span with spare ground.
    #[test]
    fn nothing_is_spare_before_a_workload_has_run() {
        let mut p = GrowthPolicy::new();
        assert_eq!(p.spare(steady(10, 500), 32, R), Err(Refusal::Observing));
    }

    /// Demand that is climbing refuses, and the refusal clears the moment the
    /// series flattens — one negotiation, not a stand-down.
    #[test]
    fn rising_demand_refuses_and_clears_when_it_flattens() {
        let mut p = GrowthPolicy::new();
        let _ = p.spare(steady(10, 500), 32, R);
        assert_eq!(p.spare(steady(20, 490), 32, R), Err(Refusal::Pressure));
        // Flat now: the same occupancy is spare.
        assert_eq!(p.spare(steady(20, 490), 32, R), Ok(490 - 32));
    }

    /// A purchase is the KV side saying it wants ground, and the very next
    /// negotiation must hear it even though occupancy looks roomy.
    #[test]
    fn a_purchase_refuses_the_next_negotiation_exactly_once() {
        let mut p = GrowthPolicy::new();
        let _ = p.spare(steady(10, 500), 32, R);
        let _ = p.spare(steady(10, 500), 32, R);
        p.note_demand();
        assert_eq!(p.spare(steady(10, 500), 32, R), Err(Refusal::Pressure));
        assert_eq!(p.spare(steady(10, 500), 32, R), Ok(500 - 32));
    }

    /// **The tier's own ground is not spare, and offering it is what churns.**
    ///
    /// Without this deduction the weight side takes ground the very next forward
    /// must buy back, and the buy-back evicts the expert slots standing on it.
    /// Measured on the 35B: 19,878 grows against 20,018 concessions in one run,
    /// ~4.6 evict-and-reload cycles a second, throughput 12,569 → ~40 tok/s.
    ///
    /// The demand guard cannot catch it — the loop hands the ground back, so
    /// demand returns to where it started and never reads as rising.
    #[test]
    fn the_next_tiers_ground_is_not_offered_as_spare() {
        let mut p = GrowthPolicy::new();
        let occ = Occupancy {
            live: 100,
            free_below_ceiling: 60,
            ceiling_blocked: 0,
            tier_bytes: 0,
            tier_high_water: 0,
            tier_planned: 19 * R,
        };
        // Two priming calls: the first is `Observing`, the second establishes a
        // flat demand so the pressure guard stands down.
        let _ = p.spare(occ, 32, R);
        let _ = p.spare(occ, 32, R);
        // 60 free, less 32 slack, less the 19 the tier is about to want.
        assert_eq!(p.spare(occ, 32, R), Ok(60 - 32 - 19));
    }

    /// A tier large enough to swallow the remaining spare leaves nothing to
    /// grant, which must read as `Occupied` rather than as a grant of zero.
    #[test]
    fn a_tier_wider_than_the_spare_refuses_outright() {
        let mut p = GrowthPolicy::new();
        let occ = Occupancy {
            live: 100,
            free_below_ceiling: 60,
            ceiling_blocked: 0,
            tier_bytes: 0,
            tier_high_water: 0,
            tier_planned: 40 * R,
        };
        let _ = p.spare(occ, 32, R);
        let _ = p.spare(occ, 32, R);
        assert_eq!(p.spare(occ, 32, R), Err(Refusal::Occupied));
    }

    /// Tier-blocked ground is offered, because a standing tier is idle between
    /// forwards and its bytes come back at phase 0.
    #[test]
    fn tier_blocked_ground_counts_as_available() {
        let mut p = GrowthPolicy::new();
        let occ = Occupancy {
            live: 100,
            free_below_ceiling: 20,
            ceiling_blocked: 80,
            tier_bytes: 0,
            tier_high_water: 57 * R,
            tier_planned: 0,
        };
        let _ = p.spare(occ, 32, R);
        let _ = p.spare(occ, 32, R);
        assert_eq!(p.spare(occ, 32, R), Ok(20 + 80 - 32));
    }

    /// A KV side genuinely full says so, and says it with the refusal that means
    /// "the partition is working" rather than "the mechanism is inert".
    #[test]
    fn a_full_kv_side_refuses_as_occupied_not_as_pressure() {
        let mut p = GrowthPolicy::new();
        let _ = p.spare(steady(500, 0), 32, R);
        let _ = p.spare(steady(500, 0), 32, R);
        assert_eq!(p.spare(steady(500, 0), 32, R), Err(Refusal::Occupied));
        // And slack is never underflowed into a grant.
        assert_eq!(p.spare(steady(500, 10), 32, R), Err(Refusal::Occupied));
    }

    /// **The ratchet, as a trajectory.** Demand rises and falls; the ground that
    /// the fall released must be offered back. Under the windowed forecast this
    /// answered zero for up to two minutes.
    #[test]
    fn ground_released_by_a_falling_cohort_is_offered_back() {
        let mut p = GrowthPolicy::new();
        // Warm up, then climb to a wide cohort.
        for live in [10usize, 60, 200, 410] {
            let _ = p.spare(steady(live, 546 - live), 32, R);
        }
        // The cohort ends; the arenas are swept and the ground is genuinely free.
        let after = steady(60, 546 - 60);
        // One negotiation absorbs the derivative flip, the next must grant.
        let _ = p.spare(after, 32, R);
        assert_eq!(
            p.spare(after, 32, R),
            Ok(546 - 60 - 32),
            "ground released by a departed cohort was not offered back"
        );
    }
}
