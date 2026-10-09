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
//! 3. **Occupancy.** What the KV side does not hold *above its highest live
//!    region*, less a slack margin — the floor moves as one edge, so a hole below
//!    a live arena is not ground the weights can take.
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

/// The free regions a KV side keeps in hand, as its scheduler defines them: below
/// [`Self::setpoint`] it is under pressure, and its relief frees to
/// [`Self::relieved`].
///
/// **One definition for both sides of the boundary.** The scheduler's relief
/// concedes weight ground until the KV side holds the relieved count free; the
/// weight side's growth takes free ground back down to its own slack. When the
/// two disagree they trade the same regions forever: on Flash-Next's 128K decode
/// the weight side left 32 regions free against a relief setpoint of 50, so every
/// decode step bought 12 regions back from the zone and the next forward's growth
/// took them again — 277 boundary moves in one Strata run, each behind two
/// device-wide quiesces. So the pool is told the target (`set_kv_free_target`) and
/// the growth policy leaves at least what relief aims for.
///
/// `divisor == 0` is no target at all: a span with no scheduler defending it, such
/// as the bare forward harness, leaves only the growth slack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FreeRegionTarget {
    /// The setpoint is the KV side's region count over this.
    pub divisor: usize,
    /// …but never fewer than this many regions.
    pub floor: usize,
    /// Regions relief frees past the setpoint, so a pass that just clears
    /// pressure does not re-trip on the next wave.
    pub overshoot: usize,
}

impl FreeRegionTarget {
    pub const fn new(divisor: usize, floor: usize, overshoot: usize) -> Self {
        Self {
            divisor,
            floor,
            overshoot,
        }
    }

    /// Free regions below which a KV side of `total` regions is under pressure:
    /// `total / divisor`, at least `floor`, and never more than half the span —
    /// on a card too small for the floor, demanding it would be permanent
    /// pressure that no relief pass could ever clear.
    pub fn setpoint(&self, total: usize) -> usize {
        if self.divisor == 0 {
            return 0;
        }
        (total / self.divisor).max(self.floor).min(total / 2)
    }

    /// The free regions relief frees to — the setpoint plus the overshoot — and
    /// so what the weight side must leave free.
    pub fn relieved(&self, total: usize) -> usize {
        match self.setpoint(total) {
            0 => 0,
            s => s + self.overshoot,
        }
    }
}

/// What the pool measures for one negotiation.
///
/// Gathered at phase 0, where the present is knowable exactly: the tier has been
/// released, no wave generation is open, and empty arenas have just been swept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Occupancy {
    /// Regions the KV side holds in all — live, free and tier-blocked.
    pub total: usize,
    /// The free regions the KV side's scheduler keeps in hand.
    pub kv_target: FreeRegionTarget,
    /// Regions held by an arena, empty or not.
    pub live: usize,
    /// Regions on the free list below the tier ceiling — claimable right now.
    pub free_below_ceiling: usize,
    /// Regions the tier's ceiling puts out of reach, plus what the weight side
    /// already holds. Idle rather than occupied: the boundary moves only with no
    /// wave open, so a standing tier's bytes are dead until phase 0 releases
    /// them.
    pub ceiling_blocked: usize,
    /// Regions between the highest live one and the top of the KV side — the
    /// frontier gap, and the only ground the weight side can actually take.
    ///
    /// The floor moves as one edge, so a free region *below* a live one is spare
    /// to the KV side and unreachable to the weights: `set_weight_floor` refuses
    /// any floor that would cut a live region. Offering holes produced a floor the
    /// pool then refused — 489 `requested boundary move failed … region 198 is
    /// live` in one night's ingest, each after a whole-device quiesce.
    pub free_above_live: usize,
    /// The transient tier's current footprint, bytes.
    pub tier_bytes: usize,
    /// The tier the most recent forward stood, bytes — what the next forward's is
    /// expected to need again, and so ground the negotiation leaves free.
    ///
    /// **The next forward's tier, by its best predictor.** A prefill runs as a run
    /// of forwards a chunk apart, each tier a little wider than the last; the
    /// negotiation between two of them used to read the first one's released ground
    /// as spare, hand it to the weight side, and the second forward's placement
    /// bought it straight back. Measured on Flash-Next's 128K prefill: a compaction
    /// between forwards lowered the live count, the negotiation granted 1,631 slots,
    /// and the next placement conceded 1,877 — twice per turn, each a pair of
    /// boundary moves. A decode's tier is a few regions, so once the prefill is done
    /// the weight side grows back into everything the prefill's tier held.
    pub last_tier_bytes: usize,
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
    /// Ground is free, but in holes below a live region — nothing the floor can
    /// reach. Compaction packs the arenas down and turns this into a gap.
    Fragmented,
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
        //
        // **So the next forward's tier is deducted, by the last forward's**
        // (`last_tier_bytes`). Deducting `transient_high_water` was tried and is
        // worse: it is the widest tier the process ever stood (up to the full
        // reservation), not the next one's price, and on the 27B it cut applied
        // grants from 17 to 4 and cost five layers of residency. The last tier is
        // the next one's price within a prefill, and a few regions once decoding.
        //
        // **What stays free besides is the slack or the KV side's relief target,
        // whichever is larger** — see [`FreeRegionTarget`]. Taking the KV side below
        // what its own relief frees to is taking ground relief will buy straight
        // back.
        let keep = slack.max(occ.kv_target.relieved(occ.total))
            + occ.last_tier_bytes.div_ceil(region_bytes.max(1));
        let by_occupancy = occ.free_below_ceiling + occ.ceiling_blocked;
        if by_occupancy.saturating_sub(keep) == 0 {
            return Err(Refusal::Occupied);
        }
        // **Only the frontier gap is takeable.** The free count includes holes
        // below live regions, which the floor cannot cross; the grant is the gap
        // less what stays free, whatever the free list says.
        match by_occupancy.min(occ.free_above_live).saturating_sub(keep) {
            0 => Err(Refusal::Fragmented),
            n => Ok(n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: usize = 16 * 1024 * 1024;

    /// A packed KV side: every free region sits above the live ones.
    fn steady(live: usize, free: usize) -> Occupancy {
        Occupancy {
            total: live + free,
            kv_target: FreeRegionTarget::default(),
            live,
            free_below_ceiling: free,
            ceiling_blocked: 0,
            free_above_live: free,
            tier_bytes: 0,
            last_tier_bytes: 0,
            tier_high_water: 0,
        }
    }

    /// **The prefill churn, closed.** Between two prefill forwards the first
    /// one's tier has been released, and its ground read as spare; the weight
    /// side took it and the next placement bought it back. The last forward's
    /// tier is left free, and only what lies beyond it is offered.
    #[test]
    fn the_last_forwards_tier_is_left_free() {
        let mut p = GrowthPolicy::new();
        let occ = Occupancy {
            last_tier_bytes: 300 * R,
            ..steady(100, 400)
        };
        let _ = p.spare(occ, 32, R);
        let _ = p.spare(occ, 32, R);
        assert_eq!(p.spare(occ, 32, R), Ok(400 - 300 - 32));
        // A tier wider than the gap leaves nothing to take.
        let wide = Occupancy {
            last_tier_bytes: 380 * R,
            ..steady(100, 400)
        };
        assert_eq!(p.spare(wide, 32, R), Err(Refusal::Occupied));
    }

    /// The relief target the scheduler defends at load: span/8, at least 24
    /// regions, freed 8 past.
    const LOAD: FreeRegionTarget = FreeRegionTarget::new(8, 24, 8);

    #[test]
    fn the_setpoint_scales_with_the_span_floors_and_clamps_to_half() {
        assert_eq!(LOAD.setpoint(800), 100, "span/8 once it clears the floor");
        assert_eq!(LOAD.setpoint(100), 24, "the floor below that");
        assert_eq!(LOAD.setpoint(32), 16, "never more than half the span");
        assert_eq!(LOAD.relieved(800), 108);
        assert_eq!(LOAD.relieved(0), 0, "no span, no demand");
        assert_eq!(FreeRegionTarget::default().relieved(800), 0, "no target");
    }

    /// **The oscillation, closed.** A weight side that took the KV side down to its
    /// 32-region slack left it below a relief setpoint of 50, and relief bought the
    /// ground straight back. With the target in the occupancy, the grant stops at
    /// what relief frees to.
    #[test]
    fn the_weight_side_leaves_what_relief_frees_to() {
        let mut p = GrowthPolicy::new();
        let occ = Occupancy {
            kv_target: LOAD,
            ..steady(330, 70)
        };
        let _ = p.spare(occ, 32, R);
        let _ = p.spare(occ, 32, R);
        // 400 regions: setpoint 50, relieved 58 — 70 free leaves 12 to take,
        // where the slack alone would have offered 38.
        assert_eq!(p.spare(occ, 32, R), Ok(70 - 58));
        // And a KV side at the relief target has nothing to give.
        let at_target = Occupancy {
            kv_target: LOAD,
            ..steady(342, 58)
        };
        let _ = p.spare(at_target, 32, R);
        assert_eq!(p.spare(at_target, 32, R), Err(Refusal::Occupied));
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
            total: 200,
            kv_target: FreeRegionTarget::default(),
            live: 100,
            free_below_ceiling: 20,
            ceiling_blocked: 80,
            free_above_live: 100,
            tier_bytes: 0,
            last_tier_bytes: 0,
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

    /// Holes below a live region are not offered: the grant is the frontier gap
    /// less the slack. The numbers are the refusal the daemon logged — 587
    /// regions, region 198 live, a free list that reached below it.
    #[test]
    fn only_the_frontier_gap_is_offered() {
        let mut p = GrowthPolicy::new();
        let occ = Occupancy {
            total: 587,
            kv_target: FreeRegionTarget::default(),
            live: 150,
            free_below_ceiling: 437,
            ceiling_blocked: 0,
            free_above_live: 587 - 199,
            tier_bytes: 0,
            last_tier_bytes: 0,
            tier_high_water: 0,
        };
        let _ = p.spare(occ, 32, R);
        let _ = p.spare(occ, 32, R);
        assert_eq!(p.spare(occ, 32, R), Ok(587 - 199 - 32));
    }

    /// Free ground that is all holes refuses as fragmented, not as occupied —
    /// the fix for one is compaction, for the other nothing.
    #[test]
    fn free_ground_below_a_live_region_refuses_as_fragmented() {
        let mut p = GrowthPolicy::new();
        let occ = Occupancy {
            total: 400,
            kv_target: FreeRegionTarget::default(),
            live: 100,
            free_below_ceiling: 300,
            ceiling_blocked: 0,
            free_above_live: 20,
            tier_bytes: 0,
            last_tier_bytes: 0,
            tier_high_water: 0,
        };
        let _ = p.spare(occ, 32, R);
        let _ = p.spare(occ, 32, R);
        assert_eq!(p.spare(occ, 32, R), Err(Refusal::Fragmented));
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
