//! Where the tier may stand and what the KV side may reach — the span's
//! arithmetic, with no device in it.
//!
//! Every function here was already a free function inside [`super::region_pool`],
//! each carrying a comment saying it had been split out *"so the rule is
//! arithmetic that tests without a device"*. They could not: `region_pool` is
//! gated on the `cuda` feature, so the rules were stranded behind exactly the
//! backend they were separated from. This module is that intent finished.
//!
//! # Why the geometry is worth isolating
//!
//! Three of the partition's worst defects were geometry, not policy:
//!
//! - A tier measured down from the weight floor landed **on top of live
//!   regions**, because a region keeps its address for the life of its arena.
//! - A placed tier made the ceiling **deaf to the boundary**, so a daemon
//!   conceded itself to its floor over thousands of retries while the ceiling
//!   answered 293 every time.
//! - Counting fresh ground from `next` alone **double-counted** regions
//!   [`claimable`] had already returned.
//!
//! None of them needed a GPU to find, and none of them were found without one.

use super::types::TARGET_ARENA_BYTES;

/// One region of the reservation.
const REGION_BYTES: usize = TARGET_ARENA_BYTES;

/// Regions `claim_region` can hand out: the free ones below the ceiling, plus
/// the fresh ones the ceiling still leaves ahead of `next`.
pub fn claimable(free_below_ceiling: usize, next: usize, ceiling: usize) -> usize {
    free_below_ceiling + ceiling.saturating_sub(next)
}

/// Regions nobody owns that the ceiling nonetheless forbids: the free ones at or
/// above it, plus everything past `next` that the ceiling has cut off.
///
/// `next.max(ceiling)` is the subtle term. Fresh ground runs from `next` to the
/// ceiling; above *both* is what neither path can reach. Taking `next` alone
/// would double-count the fresh regions [`claimable`] already returned, and
/// taking `ceiling` alone would count regions below `next` that are live.
pub fn blocked(free_at_or_above: usize, next: usize, total: usize, ceiling: usize) -> usize {
    free_at_or_above + total.saturating_sub(next.max(ceiling))
}

/// How many regions the KV side may reach, given where the tier stands.
///
/// A placed tier answers with its own base — real memory a running wave is
/// writing into, where a claim would be corruption. An absent one answers with
/// the boundary, because outside a wave nothing is standing there at all.
///
/// **A placed tier makes the ceiling deaf to the boundary.** `transient_base` is
/// an address, fixed when the tier was placed; move the weight floor and only the
/// absent arm follows it. So a tier left standing past its forward caps the KV
/// side at wherever it was put, and no concession the weight side makes can lift
/// that cap — measured, as a daemon conceding itself to its floor over thousands
/// of retries while this returned 293 every time. The tier's lifetime ending with
/// its forward is what closes it.
pub fn ceiling_regions(
    transient_base: Option<u64>,
    weight_floor: u64,
    region_base: u64,
    total: usize,
) -> usize {
    let top = match transient_base {
        Some(base) => base,
        None => weight_floor.max(region_base),
    };
    let usable = top.saturating_sub(region_base) as usize;
    (usable / REGION_BYTES).min(total)
}

/// Whether `[base, base + len)` is ground the tier may stand on, or the bytes by
/// which it is not.
///
/// **The tier may only stand on ground no arena is using.** Bounding it below by
/// the start of the KV side is not enough, and that was the defect: a region
/// already handed out keeps its address for the life of its arena, so a tier
/// measured down from the weight floor lands on top of the *highest* live regions
/// long before it reaches region zero. The region ceiling stops later claims from
/// entering the tier but cannot revoke a claim already made, so the placement is
/// where the two have to be reconciled — and the only sound answer is to refuse
/// (principle 7: refuse rather than corrupt).
pub fn tier_fits(base: u64, len: usize, live_end: u64, floor: u64) -> Result<(), u64> {
    let top = base.saturating_add(len as u64);
    // Both directions are reported, because either can be the binding one: the
    // frontier anchor computes a base upward from the arenas and overshoots
    // `floor`, while the default placement measures down from `floor` and
    // undershoots `live_end`. Taking the larger keeps the pressure figure honest
    // whichever it was — and taking a `saturating_sub` in each direction is what
    // stops the unsigned wrap the single-sided version had.
    let short = live_end.saturating_sub(base).max(top.saturating_sub(floor));
    if short > 0 {
        return Err(short);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: usize = REGION_BYTES;

    /// A tier packed against the arena frontier fits exactly, and one byte lower
    /// does not — it would be standing on a live region.
    #[test]
    fn a_tier_may_not_stand_on_a_live_region() {
        let live_end = 100 * R as u64;
        let floor = 200 * R as u64;
        assert_eq!(tier_fits(live_end, 10 * R, live_end, floor), Ok(()));
        // One byte into the arenas is a refusal, and it names the shortfall.
        assert_eq!(tier_fits(live_end - 1, 10 * R, live_end, floor), Err(1));
    }

    /// A tier that would run past the weight floor is refused by the amount it
    /// overshoots — the other direction, which the single-sided version wrapped
    /// on.
    #[test]
    fn a_tier_may_not_run_past_the_weight_floor() {
        let live_end = 100 * R as u64;
        let floor = 110 * R as u64;
        assert_eq!(tier_fits(live_end, 10 * R, live_end, floor), Ok(()));
        assert_eq!(
            tier_fits(live_end, 11 * R, live_end, floor),
            Err(R as u64),
            "overshooting the floor must report the overshoot, not wrap"
        );
    }

    /// Neither direction may wrap, at any placement — the property the
    /// `saturating_sub` pair exists for.
    #[test]
    fn the_shortfall_never_wraps_at_any_placement() {
        let floor = 50 * R as u64;
        for live in [0u64, 1, 10 * R as u64, 50 * R as u64, 200 * R as u64] {
            for base in [0u64, 1, 10 * R as u64, 60 * R as u64, 300 * R as u64] {
                for len in [0usize, 1, R, 40 * R] {
                    // The only contract: it either fits or names a positive
                    // shortfall. A wrap would surface as an absurd figure.
                    if let Err(short) = tier_fits(base, len, live, floor) {
                        assert!(short > 0);
                        assert!(
                            short <= 400 * R as u64,
                            "wrapped: base {base} len {len} live {live} → {short}"
                        );
                    }
                }
            }
        }
    }

    /// **A placed tier caps the KV side at its own base, and no boundary move
    /// lifts that cap.** The daemon that conceded itself to its floor while this
    /// answered the same number every time.
    #[test]
    fn a_placed_tier_makes_the_ceiling_deaf_to_the_boundary() {
        let region_base = 0u64;
        let total = 500;
        let tier_at = 293 * R as u64;
        let placed = |floor: u64| ceiling_regions(Some(tier_at), floor, region_base, total);
        // Whatever the weight side concedes, the answer does not move.
        assert_eq!(placed(300 * R as u64), 293);
        assert_eq!(placed(400 * R as u64), 293);
        assert_eq!(placed(500 * R as u64), 293);
        // With no tier standing it tracks the boundary, as it must.
        let absent = |floor: u64| ceiling_regions(None, floor, region_base, total);
        assert_eq!(absent(300 * R as u64), 300);
        assert_eq!(absent(400 * R as u64), 400);
    }

    /// The ceiling never exceeds the regions that exist, however far right the
    /// boundary is.
    #[test]
    fn the_ceiling_is_bounded_by_the_regions_that_exist() {
        for total in [0usize, 1, 64, 500] {
            for floor in [0u64, 10 * R as u64, 10_000 * R as u64] {
                assert!(ceiling_regions(None, floor, 0, total) <= total);
                assert!(ceiling_regions(Some(floor), floor, 0, total) <= total);
            }
        }
    }

    /// **Claimable and blocked partition the unowned ground exactly once.**
    ///
    /// The double-count this is stated against: taking `next` alone in `blocked`
    /// counts the fresh regions `claimable` has already returned, so the two sum
    /// past the regions that exist and a report claims ground twice.
    #[test]
    fn claimable_and_blocked_never_double_count() {
        for total in [0usize, 1, 16, 332, 500] {
            for next in 0..=total {
                for ceiling in 0..=total {
                    // Split the free list either side of the ceiling.
                    for free_below in 0..=3usize.min(next) {
                        for free_above in 0..=3usize {
                            let live = next.saturating_sub(free_below);
                            let c = claimable(free_below, next, ceiling);
                            let b = blocked(free_above, next, total, ceiling);
                            assert!(
                                c + b + live <= total + free_above,
                                "total {total} next {next} ceiling {ceiling}: \
                                 claimable {c} + blocked {b} + live {live} over-counts"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A ceiling at or below `next` leaves nothing fresh to claim — the boundary
    /// case where `saturating_sub` is load-bearing rather than decorative.
    #[test]
    fn a_ceiling_behind_the_watermark_offers_no_fresh_ground() {
        assert_eq!(claimable(0, 100, 100), 0);
        assert_eq!(claimable(0, 100, 40), 0);
        assert_eq!(claimable(7, 100, 40), 7, "the free list is still claimable");
        assert_eq!(claimable(0, 40, 100), 60);
    }
}
