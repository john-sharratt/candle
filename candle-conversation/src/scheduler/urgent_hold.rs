//! Background decode rows held back while a person waits on a decode.
//!
//! Every band's decode rows ride every decode step (`band`), on the premise
//! that decode is width-bound: a step's cost is its expert loads, so a row
//! beside the dialogue's costs it little. That premise fails when the rows
//! bring long contexts and the step is bound by attention instead. On
//! BlackWorld (2026-10-30), a zend dialogue decoded beside 12–13 ingest turns
//! of ~75k positions each: ~1M positions a step, 165–250 ms a step against
//! 80–110 ms at three or four rows, and the dialogue at 15–20 tokens a second.
//!
//! So while an `Urgent` turn — a person waiting on it right now — decodes or
//! prefills, `Low` rows ride only one step in [`LOW_RIDES_EVERY`]. The others
//! carry the dialogue and the bands between. The prefill counts because a
//! dialogue's tool results are its next wait: on BlackWorld (2026-10-30) each
//! tool round spent ~9 s prefilling beside 8–9 rows, with steps of 143–191 ms,
//! because the hold had ended with the decode. The hold is narrower than the
//! pause
//! `docs/continuous_fair_waves.md` §3.1 records removing, in every respect that
//! failure turned on:
//!
//! - **`Urgent` only.** npcd's characters are `High` and tick every few
//!   seconds, so a hold keyed on `High` would never release; a person's turn
//!   ends.
//! - **`Low` only.** The bands between still ride every step.
//! - **Decode rows only, and only while the `Urgent` turn is in flight.** A
//!   `Low` prefill stays the throttle's, and there is no tail after the turn.
//! - **Bounded.** A held row still rides one step in [`LOW_RIDES_EVERY`], so
//!   the background never stops.

use crate::projection::DecodePriority;

/// While an `Urgent` row decodes, `Low` rows ride one decode step in this
/// many.
///
/// Four keeps three steps in four at the dialogue's own width — on the
/// BlackWorld figures above, ~90 ms against ~220 ms — so the dialogue's
/// average step roughly halves, while the ingest still advances a step in
/// four rather than stopping.
pub const LOW_RIDES_EVERY: u32 = 4;

/// Whether this step's `Low` rows ride, counted over the run of steps an
/// `Urgent` row has been decoding in.
#[derive(Debug, Default)]
pub(super) struct UrgentHold {
    /// Steps since `Low` rows last rode, while an `Urgent` row decodes.
    held: u32,
}

impl UrgentHold {
    /// Whether this step carries `Low` rows, given whether an `Urgent` row
    /// decodes in it. Every step does when none does; beside one, every
    /// [`LOW_RIDES_EVERY`]th.
    pub fn low_rides(&mut self, urgent: bool) -> bool {
        if !urgent {
            self.held = 0;
            return true;
        }
        self.held += 1;
        if self.held >= LOW_RIDES_EVERY {
            self.held = 0;
            true
        } else {
            false
        }
    }

    /// Whether a row of `band` rides a step whose `Low` rows ride or not.
    pub fn rides(band: DecodePriority, low_rides: bool) -> bool {
        low_rides || band != DecodePriority::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_an_urgent_row_low_rides_every_step() {
        let mut h = UrgentHold::default();
        assert!((0..10).all(|_| h.low_rides(false)));
    }

    #[test]
    fn beside_an_urgent_row_low_rides_one_step_in_four() {
        let mut h = UrgentHold::default();
        let rode: Vec<bool> = (0..8).map(|_| h.low_rides(true)).collect();
        assert_eq!(
            rode,
            vec![false, false, false, true, false, false, false, true]
        );
    }

    #[test]
    fn the_count_starts_over_once_no_urgent_row_decodes() {
        let mut h = UrgentHold::default();
        assert!(!h.low_rides(true));
        assert!(!h.low_rides(true));
        assert!(h.low_rides(false));
        let rode: Vec<bool> = (0..4).map(|_| h.low_rides(true)).collect();
        assert_eq!(rode, vec![false, false, false, true]);
    }

    #[test]
    fn only_low_is_held() {
        for band in [
            DecodePriority::Normal,
            DecodePriority::High,
            DecodePriority::Urgent,
        ] {
            assert!(UrgentHold::rides(band, false), "{band:?}");
        }
        assert!(!UrgentHold::rides(DecodePriority::Low, false));
        assert!(UrgentHold::rides(DecodePriority::Low, true));
    }
}
