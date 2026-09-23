//! The two hard lines a wave may not cross, and the ground they are measured
//! against.
//!
//! **What a wave's rows are worth is [`super::rate`]'s question**, not this
//! one: an offer joins the wave while it makes the wave *faster*. This file
//! holds what is true whatever the throughput model says —
//!
//! * the **floor**: residency the engine refuses to spend, whatever it buys.
//!   An engine that streams its experts is slower at everything, including
//!   finishing the work that would give the ground back.
//! * the **midpoint**: the margin a wave carrying no decode at all must leave
//!   before starting one, so the expert working set stays alive.
//!
//! Everything the byte-fit test replaced — a setpoint, an AIMD budget, a
//! queue-length mark, an open-conversation mark, a decode-derived wave width —
//! was a proxy for the throughput question, and each was falsified on hardware
//! (`docs/wave_feeder.md` §4.5–§4.11). The rate model asks it directly; these
//! two rules are the ones no answer to it may override.

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
