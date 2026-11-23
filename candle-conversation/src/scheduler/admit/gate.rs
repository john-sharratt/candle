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
    /// The floor a **prefill** may go down to, below [`Self::zone_min`]: the
    /// expert cache's own minimum, `Scheduler::expert_zone_min_bytes`.
    ///
    /// A prefill's expert copy is once per forward and amortises over its rows, so
    /// bytes buy rows at a good rate; a decode pays per layer per step against
    /// `1 − hit(R)`, so the same byte is charged again on every step after. So a
    /// prefill may spend residency down to the point the cache can no longer
    /// serve, and no further.
    pub zone_min_prefill: u64,
    /// The tier a useful forward needs beyond what the standing wave already
    /// reserves — the next forward stands on ground the standing ground does
    /// not already hold, and no more of it. Zero when a wave is standing at
    /// least that wide.
    pub forward_reserve: u64,
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

    /// The floor a prefill stops at: the expert cache's own minimum.
    ///
    /// **No margin above it, because the minimum already is one.** What a
    /// margin here used to protect — the cache keeping something to evict, so a
    /// wave never finds every slot pinned (`Expert cache full, cannot evict (all
    /// pinned)`) — is exactly what `minimum_resident_slots` prices: the pinned
    /// set, a worst-case routed set for the layer executing, and one slot for
    /// the load that triggered the eviction. A fixed 512 MiB on top of that was
    /// residency withheld for nothing.
    pub(crate) fn prefill_floor(&self) -> u64 {
        self.zone_min_prefill
    }
}

/// The share of the model's expert set the weight zone must be unable to hold,
/// at the most it could reach, before a card counts as streaming its experts.
///
/// A card that holds nearly every expert still streams a handful — the
/// long tail the zone never quite reaches — and a prefill's residency costs its
/// decodes next to nothing there; the hold would only refuse prefills for
/// ground decode does not miss. Past this share a decode misses on a
/// significant part of every layer, and each byte a prefill takes adds to it.
pub(crate) const STREAMING_SHARE: f64 = 0.30;

/// Whether the weight zone, at the most it could reach (`achievable`), leaves
/// more than [`STREAMING_SHARE`] of the model's expert set (`set`) unheld: a
/// card that **streams** its experts.
///
/// On such a card every byte of residency a prefill takes — its K/V, and above
/// all its recurrent store, which stays for the conversation's whole life — is
/// a miss on every later decode step, so a prefill that will decode is held to
/// the decode floor whatever its band (`breaks_the_hold`). On WhiteWorld
/// (2026-11-22, Flash-Next Q3_KO: ~14.8 GB achievable against ~48 GB of
/// experts) a restart's live backlog was judged against the cache's own minimum
/// instead: 23 recurrent stores took the zone from 10.9 GB to 6.1 GB, under its
/// 7.4 GB hold, and prefill ran at ~60 t/s with 10–30% of routed experts
/// resident. A card whose zone holds all or nearly all its experts loses a
/// decode next to nothing to a prefill's residency, so there the prefill keeps
/// the cache-minimum floor and the rate model's trade alone.
///
/// A dense stack has no expert set (`set == 0`) and streams nothing.
pub(crate) fn streams_experts(achievable: u64, set: u64) -> bool {
    set > 0 && (achievable as f64) < (set as f64) * (1.0 - STREAMING_SHARE)
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
            // Under the decode hold: the expert cache's own minimum.
            zone_min_prefill: 2 << 30,
            forward_reserve: 0,
        }
    }

    /// A card streams its experts exactly when the most its zone could reach
    /// is short of the whole set; a dense stack streams nothing.
    #[test]
    fn a_card_streams_when_its_zone_cannot_hold_the_set() {
        // WhiteWorld's 3090 on Flash-Next Q3_KO, and BlackWorld's 72 GB card.
        assert!(streams_experts(14_815 << 20, 48 << 30));
        assert!(!streams_experts(60 << 30, 48 << 30));
        assert!(
            !streams_experts(48 << 30, 48 << 30),
            "a zone holding the set exactly"
        );
        // A handful streamed is not streaming: up to 30% unheld keeps the trade.
        assert!(!streams_experts(40 << 30, 48 << 30), "a sixth unheld");
        assert!(streams_experts(34 << 30, 50 << 30), "32% unheld streams");
        assert!(!streams_experts(35 << 30, 50 << 30), "exactly 30% unheld");
        assert!(!streams_experts(0, 0), "a dense stack");
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
