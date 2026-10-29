//! Rule 3 (`docs/cooperative_admission.md` §4.2): never overfill a card. A card
//! has room for a unit unless it is drawing, its slots are spoken for, or one
//! more row would take its decode step past the latency target.

use super::{capacity_of, step_with_us, Weigh};
use crate::frame::{Capacity, PLACEMENT_WIDE};
use crate::gateway::registry::MemberView;

/// The decode step past which a card is overfilled: 500 ms, two tokens a
/// second for every stream on it. Loaded cards run well above a gate's idle
/// step — run 24's BlackWorld at 364 ms with 48,918 tokens queued, run 35's
/// WhiteWorld at 208 ms for one row — and still serve; a stream slower than
/// two tokens a second is one nobody is served by.
pub const LATENCY_STEP_US: u64 = 500_000;

/// The widest a card's line is read to: rows beyond this are never one more
/// stream's worth of difference.
const WIDEST: u32 = 4096;

/// The rows beyond what it decodes now that `c` can take while its step stays
/// within [`LATENCY_STEP_US`]: the largest `n` with `step_with_us(c, n)` under
/// the target, on the line through its one-more and wide steps. `None` when the
/// line sets no limit — unlearned, or a wide step no slower than the next —
/// and zero when even one more row is past the target.
pub(crate) fn width(c: &Capacity) -> Option<u32> {
    let next = u64::from(c.decode_step_next_us);
    let wide = u64::from(c.decode_step_wide_us);
    if next == 0 || wide <= next {
        return None;
    }
    if next > LATENCY_STEP_US {
        return Some(0);
    }
    let beyond = (LATENCY_STEP_US - next) * u64::from(PLACEMENT_WIDE - 1) / (wide - next);
    let mut n = (1 + beyond).min(u64::from(WIDEST)) as u32;
    // The line rounds each step down: a row more can still be under.
    while n < WIDEST && step_with_us(c, n + 1) <= LATENCY_STEP_US {
        n += 1;
    }
    Some(n)
}

/// Whether `m`'s card has room for one more unit, a draw when `draw`.
///
/// - **Drawing**, it has none for anything but a draw: a draw holds the card's
///   admission while it runs. Draws spread by their own count (`draw_of`).
/// - **With nothing placed on it**, it has room: its own background work can
///   take its step past the target, which the gateway cannot see inside it, and
///   refusing it the first unit would leave that unit waiting on work the plan
///   is about to move away.
/// - Otherwise it has room while it is idle or has a slot its units in flight
///   have not spoken for, **and** the units handed since its last `CAPACITY`
///   and this one fit in its [`width`] less the rows the fence holds back
///   there (§4.4).
pub(crate) fn has_room(m: &MemberView, weigh: &Weigh<'_>, draw: bool) -> bool {
    let card = weigh.card(m);
    if card.drawing > 0 && !draw {
        return false;
    }
    if card.in_flight == 0 {
        return true;
    }
    let c = capacity_of(m);
    let slots = c.idle || c.slots_free > card.in_flight;
    let rows = width(c).is_none_or(|w| card.pending() < w.saturating_sub(held(w, weigh.fence(m))));
    slots && rows
}

/// The rows of a card `width` wide that a fence of `share_milli` thousandths
/// holds back for the floor, rounded up.
pub(crate) fn held(width: u32, share_milli: u32) -> u32 {
    (u64::from(width) * u64::from(share_milli.min(1000))).div_ceil(1000) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stepping(next_us: u32, wide_us: u32) -> Capacity {
        Capacity {
            decode_step_next_us: next_us,
            decode_step_wide_us: wide_us,
            ..Capacity::default()
        }
    }

    /// **The width is the most rows whose step stays within the target**, on
    /// the card's own line. Run 24's WhiteWorld — 70.5 ms widening to 451.4
    /// ms at sixteen more — stays within 500 ms for 17 rows: the 17th steps
    /// 476.8 ms, the 18th would step 502.2.
    #[test]
    fn the_width_is_the_rows_within_the_target() {
        let c = stepping(70_508, 451_420);
        assert_eq!(width(&c), Some(17));
        assert_eq!(step_with_us(&c, 17), 476_814);
        assert_eq!(step_with_us(&c, 18), 502_208);
        // Run 24's BlackWorld: 363.8 ms widening only to 420.0.
        assert_eq!(width(&stepping(363_828, 420_000)), Some(37));
    }

    /// A card already past the target takes no row by the line; a flat line,
    /// or one not learned, sets no width at all.
    #[test]
    fn past_the_target_is_no_width_and_a_flat_line_is_none() {
        assert_eq!(width(&stepping(600_000, 900_000)), Some(0));
        assert_eq!(width(&stepping(40_000, 40_000)), None, "flat");
        assert_eq!(width(&stepping(40_000, 30_000)), None, "no slower wide");
        assert_eq!(width(&stepping(0, 0)), None, "unlearned");
    }

    /// The fence's rows are its share of the width, rounded up, and never more
    /// than the width.
    #[test]
    fn the_fence_holds_back_its_share_rounded_up() {
        assert_eq!(held(17, 470), 8);
        assert_eq!(held(17, 0), 0);
        assert_eq!(held(10, 1), 1);
        assert_eq!(held(10, 1_000), 10);
        assert_eq!(held(10, 1_500), 10);
    }
}
