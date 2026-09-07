//! Whether one admission may proceed.
//!
//! Four rules, and deliberately no fifth. Everything this replaced — a byte
//! setpoint, an AIMD budget, a queue-length mark, an open-conversation mark, a
//! decode-derived wave width — was a proxy for the question below, and each was
//! falsified on hardware (`docs/wave_feeder.md` §4.5–§4.11).
//!
//! The question is: **would admitting this cost us resident experts?** An
//! engine that streams its experts is slower at everything, including finishing
//! the work that would give the ground back, so that is the one price worth
//! refusing to pay. Nothing else here is a throttle.

/// The device as the gate needs to see it, measured **after** the wave's
/// eviction pass so every figure is settled rather than forecast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Headroom {
    /// K/V bytes available without moving the boundary into the weight zone.
    /// Spending past this is spending experts.
    pub free_kv: u64,
    /// The weight zone as it stands.
    pub zone: u64,
    /// The floor it may not go under, and the residency it could reach.
    pub zone_min: u64,
    pub zone_max: u64,
}

/// Ground kept clear above the floor, so the expert cache always has something
/// it can evict for the forward it is about to run.
///
/// **The floor alone is not a safe stopping point.** Reaching it does not
/// merely shrink the cache — it leaves every remaining expert slot pinned by
/// the wave that needs it, and the forward then fails outright rather than
/// degrading: `Expert cache full, cannot evict (all pinned)`, 125 times on run
/// BT. Admission stops a slot cache's worth short of the floor so there is
/// always something to give.
const EVICTION_MARGIN: u64 = 512 << 20;

impl Headroom {
    /// Halfway between the floor and the achievable residency.
    ///
    /// The margin [`may_start_decode`] keeps the zone inside. Not a throttle —
    /// see that function for why the decode side is the thing being protected.
    pub(crate) fn midpoint(&self) -> u64 {
        self.zone_min + (self.zone_max.saturating_sub(self.zone_min)) / 2
    }

    /// The floor admission actually stops at: the hold plus the margin the
    /// cache needs to stay evictable.
    pub(crate) fn floor(&self) -> u64 {
        self.zone_min.saturating_add(EVICTION_MARGIN)
    }
}

/// Whether an admission may proceed, given what the wave already holds.
///
/// * **Nothing active — admit, whatever it costs.** A wave that admits nothing
///   frees nothing, and a slot too large to ever fit must still run rather than
///   block the queue behind it forever. This is what makes the design
///   deadlock-free, and it is why the rule is *regardless of budget* rather
///   than "against a generous budget".
/// * **Something active — admit only while it does not reach the weights.**
///   The cost is [`super::cost::Cost::total`]; the room is what eviction just
///   left. Past that the elastic boundary moves into the weight zone and the
///   admission is paid for in experts.
/// **There is deliberately no third clause about the zone standing under its
/// floor.** That reads as the obvious safety rail and behaves as a wedge: the
/// zone settles just under the floor and stays there, so a gate keyed on it
/// refuses essentially every admission and the engine only ever drains. Run BY
/// measured the cost — effective 5,158 MiB against a 5,284 MiB floor, the queue
/// never draining past four, two directories in ten minutes. A shortfall is a
/// *debt*, and `WaveFill::headroom` prices it against the free list so admission
/// throttles in proportion to it. Keep that here and this stays one question.
pub(crate) fn may_admit(active: usize, cost: u64, h: &Headroom) -> bool {
    active == 0 || cost <= h.free_kv
}

/// Whether a wave carrying no decode at all may start one.
///
/// **This exists to keep the decode weights hot.** Prefill routes across many
/// experts; with no decode resident to hold its working set alive, a run of
/// prefills mass-evicts the experts the next decode needs, and that decode then
/// pays to stream every one of them back. Keeping one decode alive is what
/// keeps the cache warm.
///
/// So this is checked against the **midpoint**, not the floor: the decode is
/// admitted only while the zone stays in the upper half of its range, which is
/// the margin that keeps the working set alive rather than merely legal.
///
/// **It must only ever fire when no decode is active.** Gating the second and
/// subsequent decodes on a weight point is the wedge this engine was rebuilt to
/// remove: it froze 547 of ~900 fills at one decode a wave, and the zone could
/// never recover, because recovery needs completions and completions need
/// decodes. If a later tidy-up makes this general, that is the failure it will
/// reintroduce.
pub(crate) fn may_start_decode(decodes_active: usize, cost: u64, h: &Headroom) -> bool {
    decodes_active == 0 && h.zone.saturating_sub(cost) > h.midpoint()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn room(free: u64) -> Headroom {
        Headroom {
            free_kv: free,
            zone: 8 << 30,
            zone_min: 4 << 30,
            zone_max: 10 << 30,
        }
    }

    /// The rule that makes the design deadlock-free: an empty wave takes the
    /// head whatever it costs, so a slot too large to fit still runs.
    #[test]
    fn an_empty_wave_admits_regardless_of_cost() {
        assert!(may_admit(0, u64::MAX, &room(0)));
    }

    #[test]
    fn a_busy_wave_admits_only_what_the_free_ground_covers() {
        let h = room(1_000);
        assert!(may_admit(3, 1_000, &h), "exactly the room is still room");
        assert!(!may_admit(3, 1_001, &h), "one byte past is the weights");
        assert!(may_admit(3, 0, &h));
    }

    /// **A sunk zone does not close the gate here**, and the guard is against
    /// re-adding one: this reads like the obvious safety rail, and run BY showed
    /// it wedges the engine at two directories in ten minutes because the zone
    /// settles just under its floor and never climbs back out on its own. The
    /// shortfall is priced against the free list in `WaveFill::headroom`, so by
    /// the time a `Headroom` reaches this function the debt is already paid and
    /// `free_kv` is the whole answer.
    #[test]
    fn a_sunk_zone_is_not_this_functions_business() {
        let sunk = Headroom {
            free_kv: 1_000,
            zone: (4 << 30) + EVICTION_MARGIN - 1,
            zone_min: 4 << 30,
            zone_max: 10 << 30,
        };
        assert!(sunk.zone < sunk.floor(), "the zone is under its floor");
        assert!(
            may_admit(3, 1_000, &sunk),
            "room already net of the debt is room, whatever the zone reads",
        );
        assert!(
            !may_admit(3, 1_001, &sunk),
            "and the free ground is still the bound",
        );
    }

    #[test]
    fn the_midpoint_is_halfway_between_the_floor_and_the_achievable() {
        assert_eq!(room(0).midpoint(), 7 << 30);
    }

    /// **Admission stops above the hold, not at it.** At the hold the expert
    /// cache has nothing left to evict and the forward fails outright rather
    /// than running slower — `cannot evict (all pinned)`, 125 forwards on run
    /// BT. The margin is what keeps it evictable.
    #[test]
    fn the_floor_keeps_a_margin_above_the_hold() {
        let h = room(0);
        assert!(h.floor() > h.zone_min, "the floor must clear the hold");
        assert_eq!(h.floor(), (4 << 30) + EVICTION_MARGIN);
        assert!(
            h.floor() < h.zone_max,
            "and still leave a range to admit into",
        );
    }

    /// Only when nothing is decoding, and only with the zone left in its upper
    /// half — the margin that keeps the expert working set alive.
    #[test]
    fn a_first_decode_starts_only_with_the_zone_in_its_upper_half() {
        let h = room(u64::MAX);
        assert!(
            may_start_decode(0, 0, &h),
            "8 GiB clears the 7 GiB midpoint"
        );
        assert!(
            !may_start_decode(0, 2 << 30, &h),
            "a cost that drops the zone to the midpoint does not start one",
        );
        assert!(
            !may_start_decode(1, 0, &h),
            "a decode is already keeping the cache warm",
        );
    }
}
