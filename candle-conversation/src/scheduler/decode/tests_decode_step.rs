//! Tests the rollback targets of a speculative step.
//!
//! A row that committed nothing is excluded from the rollback. A rewind to zero
//! rows is refused, and asking for one would fail every decode in the step, so the
//! row is left out. The rows that did commit roll back to just past their last
//! committed token.

use super::*;

#[cfg(test)]
pub(super) mod rollback_target_tests {
    use super::rollback_targets;

    /// A row that committed nothing stays out of the rollback; the others
    /// roll back to just past what they committed.
    #[test]
    fn a_row_that_committed_nothing_is_left_out_of_the_rollback() {
        let (stopped, targets) = rollback_targets(&[21, 26, 30], &[100, 200, 300], &[3, 0, 1]);
        assert_eq!(stopped, vec![1]);
        assert_eq!(targets, vec![(21, 103), (30, 301)]);
        let (stopped, targets) = rollback_targets(&[5], &[10], &[2]);
        assert!(stopped.is_empty());
        assert_eq!(targets, vec![(5, 12)]);
    }
}
