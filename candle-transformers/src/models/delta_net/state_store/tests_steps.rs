//! Tests the step rules: an undone step returns every committed segment to its
//! entry state, and a kept step folds what its commits owed.
//!
//! A segment that fails partway leaves no trace, and a partial sweep that commits
//! advances only the layers it reached. The armed store is owed no fold and keeps
//! its rows for the reconcile. A step's undo is refused when the state moved by an
//! exchange or a fold rather than by a short span.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_layers::tests::{filled_store, long_span, short_span};

    /// **The close folds the stores it is told to and leaves the armed one
    /// pending**, then commits both: a plain row's rows land at its own commit,
    /// a verifying sequence's wait for the reconcile.
    #[test]
    fn commit_stores_folds_the_unarmed_and_leaves_the_armed_pending() {
        let mut plain = filled_store();
        let mut armed = filled_store();
        let entry = plain.export().unwrap();
        for store in [&mut plain, &mut armed] {
            store.begin_wave().unwrap();
            short_span(store, 0, 2.0, 7.0);
        }
        commit_stores(&mut [(&mut plain, true), (&mut armed, false)]).unwrap();
        assert!(!plain.has_pending(), "the plain row folded at its commit");
        assert!(
            armed.has_pending(),
            "the armed block waits for its reconcile"
        );
        assert_ne!(plain.export().unwrap()[0], entry[0]);
        // Both waves are closed: a new one opens on either.
        plain.begin_wave().unwrap();
        plain.rollback_wave().unwrap();
        armed.begin_wave().unwrap();
        armed.rollback_wave().unwrap();
        assert!(
            armed.has_pending(),
            "a later wave's rollback keeps the armed rows"
        );
    }

    /// **A failed long-span wave that reached only part of the stack leaves NO
    /// trace**, and the next wave does not inherit its bookkeeping: a commit that
    /// exchanged layers the dead wave touched would install its output.
    #[test]
    fn a_partial_sweep_that_rolls_back_leaves_every_layer_at_its_entry_value() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        let entry = store.export().unwrap();

        store.begin_wave().unwrap();
        for li in [0usize, 1] {
            long_span(&mut store, li);
        }
        store.rollback_wave().unwrap();
        assert_eq!(
            store.export().unwrap(),
            entry,
            "a rolled-back partial sweep moved the state"
        );

        store.begin_wave().unwrap();
        store.commit_wave();
        assert_eq!(
            store.export().unwrap(),
            entry,
            "a later commit installed the FAILED wave's output — `advanced` \
             outlived the rollback"
        );
    }

    /// A partial sweep that COMMITS advances exactly the layers it reached, and
    /// leaves the rest alone.
    #[test]
    fn a_partial_sweep_that_commits_advances_only_the_layers_it_reached() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        let entry = store.export().unwrap();

        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        store.commit_wave();

        let after = store.export().unwrap();
        assert_ne!(after[0].state, entry[0].state, "layer 0 advanced");
        assert_eq!(
            after[1].state, entry[1].state,
            "layer 1 was never reached and must not have moved"
        );
        assert_eq!(after[2].state, entry[2].state, "nor layer 3");
    }

    /// **A step whose later segment fails is undone whole.** Segment one
    /// commits layers 0 and 1 — their rows stay pending, the step folding
    /// nothing — and segment two is refused on layer 3: undoing the step drops
    /// the committed rows, the state never having moved, so the step's retry
    /// runs their tokens into the state once rather than twice.
    #[test]
    fn an_undone_step_returns_every_committed_segment_to_its_entry_state() {
        let mut store = filled_store();
        let entry = store.export().unwrap();

        store.begin_step().unwrap();
        store.begin_wave().unwrap();
        for layer in [0, 1] {
            short_span(&mut store, layer, 2.0, 7.0);
        }
        commit_stores(&mut [(&mut store, true)]).unwrap();
        assert!(
            store.has_pending(),
            "inside a step the commit folds nothing"
        );
        store.begin_wave().unwrap();
        short_span(&mut store, 3, 2.0, 7.0);
        store.rollback_wave().unwrap();

        store.undo_step().unwrap();
        assert!(!store.has_pending());
        assert_eq!(store.export().unwrap(), entry, "every layer back at entry");
        // The retry runs the step's layers again.
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 2.0, 7.0);
        store.rollback_wave().unwrap();
    }

    /// **A kept step folds at its end what its commits owed**, in one fold
    /// over every store; an armed store was owed none and stays pending for
    /// its reconcile.
    #[test]
    fn a_kept_step_folds_what_its_commits_owed() {
        let mut plain = filled_store();
        let mut armed = filled_store();
        let entry = plain.export().unwrap();
        for store in [&mut plain, &mut armed] {
            store.begin_step().unwrap();
        }
        for layer in [0, 1] {
            for store in [&mut plain, &mut armed] {
                store.begin_wave().unwrap();
                short_span(store, layer, 2.0, 7.0);
            }
            commit_stores(&mut [(&mut plain, true), (&mut armed, false)]).unwrap();
        }
        assert_eq!(
            plain.pending_layers().collect::<Vec<_>>(),
            vec![(0, 1), (1, 1)],
            "nothing folded mid-step"
        );
        end_steps(&mut [&mut plain, &mut armed]).unwrap();
        assert!(!plain.has_pending());
        let after = plain.export().unwrap();
        assert_ne!(after[0], entry[0]);
        assert_ne!(after[1], entry[1]);
        assert_eq!(after[2], entry[2], "layer 3 never ran");
        assert_eq!(
            armed.pending_layers().collect::<Vec<_>>(),
            vec![(0, 1), (1, 1)],
            "the armed rows wait for the reconcile"
        );
        // Both steps closed: a commit folds again.
        plain.begin_wave().unwrap();
        short_span(&mut plain, 3, 2.0, 7.0);
        commit_stores(&mut [(&mut plain, true)]).unwrap();
        assert!(!plain.has_pending());
    }

    /// Rows pending when the step began are not the step's: an undo keeps them
    /// and drops only what the step's waves wrote.
    #[test]
    fn a_step_undo_keeps_the_rows_it_entered_with() {
        let mut store = filled_store();
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 2.0, 7.0);
        commit_stores(&mut [(&mut store, false)]).unwrap();

        store.begin_step().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 2.0, 7.0);
        commit_stores(&mut [(&mut store, false)]).unwrap();
        store.undo_step().unwrap();
        assert_eq!(store.pending_layers().collect::<Vec<_>>(), vec![(0, 1)]);
    }

    /// A step whose state moved other than by a short span refuses its undo
    /// instead of dropping rows from a state they no longer describe — an
    /// exchange, or a fold — and the refusal closes the step. A layer the step
    /// already ran refuses a second span outright.
    #[test]
    fn a_step_undo_refuses_what_dropping_rows_cannot_restore() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        store.begin_step().unwrap();
        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        commit_stores(&mut [(&mut store, true)]).unwrap();
        assert!(store.undo_step().is_err(), "a long span's exchange");
        assert!(
            store.undo_step().is_err(),
            "the refused undo closed the step"
        );

        store.begin_step().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 2.0, 7.0);
        commit_stores(&mut [(&mut store, false)]).unwrap();
        store.fold(None).unwrap();
        assert!(store.undo_step().is_err(), "a fold inside the step");

        store.begin_step().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 2.0, 7.0);
        commit_stores(&mut [(&mut store, true)]).unwrap();
        store.begin_wave().unwrap();
        assert!(
            store.layer_span_mut(1, 1).is_err(),
            "a layer the step ran holds its rows"
        );
        store.rollback_wave().unwrap();
        store.undo_step().unwrap();
        assert!(!store.has_pending());
    }
}
