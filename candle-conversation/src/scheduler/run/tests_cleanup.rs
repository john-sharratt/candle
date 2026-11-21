//! Tests for when a finished turn's cleanup is due.
//!
//! Nothing finished means nothing is due, however long the quantum has run. A
//! cohort that has drained cleans up at once. A finished turn that still has
//! other turns decoding waits up to the bound and no longer. A turn that wrote a
//! tool call is cleaned up in the step it finishes, because its caller is waiting
//! on the `Done` to run the tool.

use super::*;

#[cfg(test)]
pub(super) mod cleanup_deferral_tests {
    use super::{cleanup_due, CLEANUP_DEFER};

    use std::time::{Duration, Instant};

    /// Nothing finished, nothing to clean up — however long the quantum has run.
    #[test]
    fn nothing_finished_is_never_due() {
        assert!(!cleanup_due(None, Instant::now(), 0, false));
        assert!(!cleanup_due(None, Instant::now(), 8, false));
        assert!(!cleanup_due(None, Instant::now(), 8, true));
    }

    /// The cohort drained: clean up at once, without waiting out the bound.
    #[test]
    fn a_drained_cohort_cleans_up_immediately() {
        let now = Instant::now();
        assert!(cleanup_due(Some(now), now, 0, false));
    }

    /// Others still decoding: the finished turn waits until the bound, and no longer.
    #[test]
    fn a_finished_turn_waits_for_the_others_up_to_the_bound() {
        let at = Instant::now();
        assert!(!cleanup_due(
            Some(at),
            at + Duration::from_millis(40),
            6,
            false
        ));
        let just_short = at + CLEANUP_DEFER - Duration::from_millis(1);
        assert!(!cleanup_due(Some(at), just_short, 6, false));
        assert!(cleanup_due(Some(at), at + CLEANUP_DEFER, 6, false));
    }

    /// A turn that wrote a tool call is cleaned up the step it finishes, with
    /// others still decoding: its caller is waiting on the `Done` to run the tool.
    #[test]
    fn a_tool_call_turn_is_cleaned_up_without_waiting() {
        let at = Instant::now();
        assert!(cleanup_due(Some(at), at, 6, true));
    }
}
