//! The slots an admission pass walked work past on with no place in its group
//! (`admit_ground::AdmitPass::peek_prefill`), and what that holds back.
//!
//! Work does one of two things to its slot: a turn runs on it, so changes it;
//! a turn-group case or a submission forks a view of it, so reads it. Later
//! work on a slot waits too where going ahead would reorder the two —
//! anything behind a turn, and a turn behind anything — so no turn changes a
//! slot before work queued ahead of it has read it, and no fork reads a slot a
//! turn ahead of it has yet to change. Forks of one slot are independent of
//! each other.

use std::collections::HashSet;

/// The slots work was walked past on, by what it does to the slot.
#[derive(Default)]
pub(super) struct Walked {
    turns: HashSet<usize>,
    forks: HashSet<usize>,
}

impl Walked {
    /// Whether work on `slot` — a `fork` of it, or a turn on it — waits behind
    /// what was walked past there.
    pub(super) fn holds(&self, slot: usize, fork: bool) -> bool {
        self.turns.contains(&slot) || (!fork && self.forks.contains(&slot))
    }

    /// Work on `slot` — a `fork` of it, or a turn on it — was walked past,
    /// whether it had no place or was itself held.
    pub(super) fn past(&mut self, slot: usize, fork: bool) {
        if fork {
            self.forks.insert(slot);
        } else {
            self.turns.insert(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Walked;

    /// **Walked-past work holds back only what it would be reordered with.**
    /// A turn on a slot holds everything after it there; a fork — a case or
    /// a submission — holds only a later turn, and its sibling forks go ahead.
    #[test]
    fn walked_work_holds_only_what_it_orders_with() {
        let mut walked = Walked::default();
        walked.past(1, true);
        assert!(
            !walked.holds(1, true),
            "a sibling fork reads the same state"
        );
        assert!(walked.holds(1, false), "a turn would change what it reads");
        assert!(!walked.holds(2, false), "another slot");
        walked.past(1, false);
        assert!(
            walked.holds(1, true),
            "a fork would read the state too early"
        );
    }

    /// **Held work holds what is behind it.** A fork walked past holds a
    /// turn; that turn, recorded as walked itself, then holds a later fork,
    /// which would otherwise read the slot before the turn changed it.
    #[test]
    fn held_work_holds_what_is_behind_it() {
        let mut walked = Walked::default();
        walked.past(1, true);
        assert!(walked.holds(1, false));
        walked.past(1, false);
        assert!(walked.holds(1, true));
    }
}
