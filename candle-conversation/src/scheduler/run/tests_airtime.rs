//! Tests for the creep's airtime debt: what it owes decode, when it is paid, and
//! when it owes nothing.
//!
//! The cases follow the measured ones. A creep that rides a high-ratio decode owes
//! far more decode-only airtime than one riding a low-ratio decode, a carrying
//! step sets its debt to its own share rather than adding to what was owed, and a
//! step no longer than a decode-only one owes nothing. The decode-only average
//! starts at the first step and moves toward each new one.

use super::*;

#[cfg(test)]
pub(super) mod decode_airtime_tests {
    use super::{creep_owed_after, next_creep_owed, next_decode_only};

    use std::time::Duration;

    pub(in super::super) fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// **The creep rides again once decode is paid, a step or two later.** At
    /// `R = 1` a 120 ms co-batched step beside 40 ms decode-only steps owes its
    /// 80 ms share; two decode-only steps pay it and the third may carry the
    /// creep — where it used to wait for the next 2 s quantum.
    #[test]
    fn decode_only_steps_pay_the_debt_and_free_the_creep() {
        let mut owed = next_creep_owed(Duration::ZERO, ms(120), true, Some(ms(40)), 1.0);
        assert_eq!(owed, ms(80));
        owed = next_creep_owed(owed, ms(40), false, Some(ms(40)), 1.0);
        assert!(!owed.is_zero(), "still owed");
        owed = next_creep_owed(owed, ms(40), false, Some(ms(40)), 1.0);
        assert!(owed.is_zero(), "paid: the next step carries the creep");
    }

    /// **The measured case.** A ~30 ms dialogue decode carrying one layer of an
    /// ingest group took 70 ms (`.5`, 2026-10-17). Behind that `High` decode
    /// (`R = 64`) the 40 ms share owes 2.56 s of decode-only airtime: about 85
    /// decode steps before the ingest rides again, not the next one.
    #[test]
    fn a_throttled_creep_owes_its_share_times_the_ratio() {
        let owed = next_creep_owed(Duration::ZERO, ms(70), true, Some(ms(30)), 64.0);
        assert_eq!(owed, ms(2_560));
        let mut left = owed;
        let mut steps = 0;
        while !left.is_zero() {
            left = next_creep_owed(left, ms(30), false, Some(ms(30)), 64.0);
            steps += 1;
        }
        assert_eq!(steps, 86);
    }

    /// A step that carried the creep sets the debt to its own share rather than
    /// adding to what was owed: it only rode because the last debt was paid.
    #[test]
    fn a_carrying_step_sets_the_debt_to_its_own_share() {
        assert_eq!(
            next_creep_owed(ms(500), ms(300), true, Some(ms(100)), 1.0),
            ms(200)
        );
        assert_eq!(
            next_creep_owed(ms(500), ms(300), true, Some(ms(100)), 16.0),
            ms(3_200)
        );
    }

    /// The measured case: a 4.2 s co-batched step under a `Low` decode (R = 1)
    /// owes the decode its own share — equal airtime.
    #[test]
    fn an_unthrottled_creep_owes_the_decode_its_share() {
        assert_eq!(creep_owed_after(ms(4_200), Some(ms(200)), 1.0), ms(4_000));
    }

    /// Before any decode-only step has been measured the whole step counts as
    /// the creep's: the error is toward decode.
    #[test]
    fn with_no_decode_only_step_measured_the_whole_step_is_owed() {
        assert_eq!(creep_owed_after(ms(70), None, 4.0), ms(280));
    }

    /// A step no longer than a decode-only one owes nothing — the creep cost
    /// decode no airtime — and neither does a zero ratio: a split whose slice
    /// already carries decode's share (`turn_airtime::debt_ratio`). A band's
    /// ratio is at least 1 before it gets here.
    #[test]
    fn a_free_ride_owes_nothing_and_neither_does_a_zero_ratio() {
        assert!(creep_owed_after(ms(25), Some(ms(30)), 64.0).is_zero());
        assert!(creep_owed_after(ms(3_000), Some(ms(1_000)), 0.0).is_zero());
    }

    /// The decode-only average starts at the first step and moves a quarter of
    /// the way to each new one.
    #[test]
    fn the_decode_only_average_follows_the_steps() {
        let a = next_decode_only(None, ms(40));
        assert_eq!(a, ms(40));
        let a = next_decode_only(Some(a), ms(80));
        assert_eq!(a, ms(50));
    }
}
