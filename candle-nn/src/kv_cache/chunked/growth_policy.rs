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
/// full offer overshot, the KV side bought the ground straight back through
/// `set_ground_broker`, and the purchase set the pressure guard that refuses the
/// *next* negotiation. Applied grants fell from four to two and the zone settled
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
    /// **Being short of KV no longer fails a forward.** A claim that runs the KV
    /// side out buys exactly the ground it needs at the moment it needs it
    /// (`set_ground_broker` → `sell_ground`), and the weight side concedes on
    /// contact. Measured over a full 27B gate: thirty purchases, **zero
    /// refused**. The forecast was insurance against a loss that can no longer
    /// occur, and it was not free.
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
        // statement is that this offers ground the tier may then have to buy
        // back through `set_ground_broker`, at the cost of churn rather than
        // failure.
        //
        // Deducting `transient_high_water` was tried and is worse: it is the
        // widest tier the process ever stood (up to the full reservation), not
        // the next one's price, and on the 27B it cut applied grants from 17 to 4
        // and cost five layers of residency. The right term is this wave's
        // planned tier, which `WavePlan` knows and this signature does not — so
        // it is left undeducted deliberately, and named.
        let by_occupancy = occ.free_below_ceiling + occ.ceiling_blocked;
        match by_occupancy.saturating_sub(slack) {
            0 => Err(Refusal::Occupied),
            n => Ok(n),
        }
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
