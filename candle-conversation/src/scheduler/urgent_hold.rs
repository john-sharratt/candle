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

/// While an `Urgent` turn is in flight, a `Low` row rides one decode step in
/// this many.
///
/// Four keeps three steps in four at the dialogue's own width — on the
/// BlackWorld figures above, ~90 ms against ~220 ms — so the dialogue's
/// average step roughly halves, while the ingest still advances a step in
/// four rather than stopping.
pub const LOW_RIDES_EVERY: u32 = 4;

/// Which `Low` rows ride this step, counted over the steps an `Urgent` turn has
/// been in flight.
///
/// **Each row on a step of its own.** All of them riding one step in four made
/// that step the whole cost: 250–420 ms beside 8–10 rows against ~60 ms for the
/// three around it, a stall in every stream beside it. A row's own id picks its
/// step, so the rows spread across the four and each still rides exactly one.
#[derive(Debug, Default)]
pub(super) struct UrgentHold {
    /// Steps of the current run; `None` while no `Urgent` turn is in flight.
    step: Option<u64>,
}

impl UrgentHold {
    /// Start a step, given whether an `Urgent` turn is in flight in it.
    pub fn advance(&mut self, urgent: bool) {
        self.step = urgent.then(|| self.step.map_or(1, |s| s + 1));
    }

    /// Whether this step carries sequence `seq` of `band`.
    pub fn rides(&self, seq: usize, band: DecodePriority) -> bool {
        match self.step {
            Some(step) if band == DecodePriority::Low => {
                (seq as u64 + step).is_multiple_of(u64::from(LOW_RIDES_EVERY))
            }
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use DecodePriority::Low;

    /// The steps of `n` on which each of `seqs` rode.
    fn rode(seqs: &[usize], n: usize) -> Vec<Vec<bool>> {
        let mut h = UrgentHold::default();
        let mut out = vec![Vec::new(); seqs.len()];
        for _ in 0..n {
            h.advance(true);
            for (i, &s) in seqs.iter().enumerate() {
                out[i].push(h.rides(s, Low));
            }
        }
        out
    }

    #[test]
    fn without_an_urgent_turn_low_rides_every_step() {
        let mut h = UrgentHold::default();
        for s in 0..10 {
            h.advance(false);
            assert!(h.rides(s, Low));
        }
    }

    #[test]
    fn beside_an_urgent_turn_a_low_row_rides_one_step_in_four() {
        assert_eq!(
            rode(&[3], 8),
            vec![vec![true, false, false, false, true, false, false, false]]
        );
    }

    /// **Rows spread across the steps.** Four rows of different residues never
    /// share one, so no step carries them all.
    #[test]
    fn low_rows_ride_on_steps_of_their_own() {
        let rode = rode(&[0, 1, 2, 3], 8);
        for step in 0..8 {
            assert_eq!(rode.iter().filter(|r| r[step]).count(), 1, "step {step}");
        }
    }

    #[test]
    fn the_count_starts_over_once_no_urgent_turn_is_in_flight() {
        let mut h = UrgentHold::default();
        h.advance(true);
        h.advance(true);
        h.advance(false);
        h.advance(true);
        assert!(h.rides(3, Low), "the first step of a new run");
    }

    #[test]
    fn only_low_is_held() {
        let mut h = UrgentHold::default();
        h.advance(true);
        for band in [
            DecodePriority::Normal,
            DecodePriority::High,
            DecodePriority::Urgent,
        ] {
            assert!(h.rides(0, band), "{band:?}");
        }
        assert!(!h.rides(0, Low));
    }
}
