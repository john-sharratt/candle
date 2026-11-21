//! Tests the rules for when a fork or a first turn may inherit or swap a prompt's
//! recurrent memory.
//!
//! Each rule is asserted on its predicate, since an end-to-end check costs two
//! model loads and a wrong memory still yields correct key-value state and fluent
//! answers that recall probes do not distinguish. A fork inherits the parent's
//! memory only on the parent's own timeline. A dialed first turn swaps the prompt
//! checkpoint only when it names a different branch, and a conversation with no
//! checkpoint builds one.

use super::*;

#[cfg(test)]
pub(super) mod branch_swap_tests {
    use super::needs_branch_swap;
    use crate::persistence::content_hash::ContentHash;

    const A: ContentHash = ContentHash { lo: 1, hi: 2 };
    const B: ContentHash = ContentHash { lo: 3, hi: 4 };

    /// Dials that name the branch already installed — the thinking switch on a
    /// prompt without section trees — swap nothing.
    #[test]
    fn a_selection_naming_the_installed_branch_swaps_nothing() {
        assert!(!needs_branch_swap(Some(A), A));
    }

    /// Dials that name a different branch swap it, which is what the window exists for.
    #[test]
    fn a_selection_naming_another_branch_swaps() {
        assert!(needs_branch_swap(Some(A), B));
    }

    /// No checkpoint went in at create — a failed pass — so the first turn builds one.
    #[test]
    fn a_conversation_with_no_installed_checkpoint_swaps() {
        assert!(needs_branch_swap(None, A));
    }
}

/// **A1 — a fork inherits the parent's memory only on the parent's own
/// timeline.**
///
/// Recurrent memory describes a token history. A fork's spliced K/V comes from
/// its *target* timeline's sealed records, so its memory must come from there
/// too; the parent's live state is the right answer only when the fork lands
/// back where the parent already is.
///
/// This is asserted here, on the predicate, rather than only end-to-end because
/// the end-to-end oracle costs two model loads and fourteen minutes, and because
/// nothing else can see the defect: a conversation given the wrong memory holds
/// the *right* K/V, answers from it fluently, and passes every recall probe.
#[cfg(test)]
pub(super) mod fork_inherits_history_tests {
    use super::{fork_inherits_history, InheritsHistory};
    use crate::projection::TimelineId;

    #[test]
    fn only_a_fork_onto_the_parents_own_timeline_takes_the_parents_memory() {
        let parent = TimelineId::for_test(7);

        // Re-attaching to the timeline the parent is already on: its live state
        // is current, describes exactly this history, and beats the snapshot.
        assert_eq!(
            fork_inherits_history(parent, parent),
            InheritsHistory::Yes,
            "a fork onto the parent's own timeline continues the parent's turns"
        );

        // The daemon's resume: fork the *base* conversation onto a client's
        // timeline. The base holds a memory of turns this client never had, so
        // the snapshot `create_sequence` restored must stand. Deciding this per
        // constructor instead is how every restored conversation came to run on
        // the base conversation's memory.
        assert_eq!(
            fork_inherits_history(parent, TimelineId::for_test(9)),
            InheritsHistory::No,
            "resuming another timeline must keep that timeline's restored snapshot"
        );

        // A freshly minted timeline — `fork` and `fork_scope` — holds nothing.
        assert_eq!(
            fork_inherits_history(parent, TimelineId::for_test(1)),
            InheritsHistory::No
        );
    }
}

/// See [`fork_wants_branch_checkpoint`]'s doc for why each of the three
/// conditions is asserted independently: the failure mode a wrong one produces
/// is either wasted compute (recomputing what `fork_recurrent` already carried)
/// or silent corruption (discarding a real conversation's memory) — neither of
/// which any end-to-end recall probe distinguishes from correct behaviour.
#[cfg(test)]
pub(super) mod fork_wants_branch_checkpoint_tests {
    use super::{fork_wants_branch_checkpoint, InheritsHistory};

    #[test]
    fn only_a_fresh_timeline_on_a_recurrent_model_with_no_turns_wants_it() {
        assert!(fork_wants_branch_checkpoint(InheritsHistory::No, true, 0));
    }

    #[test]
    fn a_fork_that_continues_the_parents_own_history_does_not() {
        // `fork_recurrent` already carries the live parent's state across for
        // this case — installing the checkpoint on top would just redo it.
        assert!(!fork_wants_branch_checkpoint(InheritsHistory::Yes, true, 0));
    }

    #[test]
    fn a_plain_transformer_never_wants_it() {
        // No checkpoint was ever computed for this model, at any turn count —
        // the lookup would only ever miss.
        assert!(!fork_wants_branch_checkpoint(InheritsHistory::No, false, 0));
    }

    #[test]
    fn resuming_a_timeline_that_already_has_turns_does_not() {
        // This is the daemon's client-resume case: `InheritsHistory::No` (a
        // fork onto some OTHER conversation's timeline) with real history
        // already sealed there. `create_sequence_seeded` already restored
        // that timeline's own recurrent snapshot; installing the prompt-only
        // checkpoint on top would silently discard it.
        assert!(!fork_wants_branch_checkpoint(InheritsHistory::No, true, 3));
    }
}
