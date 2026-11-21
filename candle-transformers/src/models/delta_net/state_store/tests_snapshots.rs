//! Tests the export and import round trip, and the seeding that keeps a restored
//! or forked state from being reset.
//!
//! A seal, a drop and a resume restore a bit-identical state, and a snapshot under
//! a foreign schedule is refused without changing the store. A fork or a restore is
//! seeded exactly once, so its first wave keeps the state, and a refused import
//! seeds nothing.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_layers::tests::{dims, filled_store, kinds, long_span, short_span};

    #[test]
    fn export_import_roundtrips_exactly() {
        let store = filled_store();
        let hash = store.schedule_hash();
        let exported = store.export().unwrap();
        assert_eq!(exported.len(), 3);
        assert_eq!(exported[2].layer_index, 3);

        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        fresh.import(hash, &exported).unwrap();
        let re = fresh.export().unwrap();
        assert_eq!(exported, re, "export→import→export must be byte-identical");
    }

    #[test]
    fn import_refuses_wrong_hash_and_wrong_geometry() {
        let store = filled_store();
        let exported = store.export().unwrap();

        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        let before = fresh.export().unwrap();
        let err = fresh
            .import(store.schedule_hash() ^ 1, &exported)
            .unwrap_err();
        assert!(err.to_string().contains("schedule hash"));
        assert_eq!(
            fresh.export().unwrap(),
            before,
            "refusal must not touch state"
        );

        let mut bad = exported.clone();
        bad[0].d_k = 5;
        let err = fresh.import(store.schedule_hash(), &bad).unwrap_err();
        assert!(err.to_string().contains("geometry"));
        assert_eq!(fresh.export().unwrap(), before);
    }

    /// A snapshot is checked against the geometry with no store built: the
    /// store's own export passes, and a wrong hash, a wrong layer count and a
    /// wrong layer geometry are each refused — the same answers `import` gives.
    #[test]
    fn check_snapshot_agrees_with_import_without_a_store() {
        let store = filled_store();
        let hash = store.schedule_hash();
        let exported = store.export().unwrap();
        let check = |h: u64, l: &[ExportedLayerState]| {
            RecurrentStateStore::check_snapshot(&kinds(), &dims(), h, l)
        };
        check(hash, &exported).unwrap();
        assert!(check(hash ^ 1, &exported)
            .unwrap_err()
            .to_string()
            .contains("schedule hash"));
        assert!(check(hash, &exported[..2])
            .unwrap_err()
            .to_string()
            .contains("snapshot has 2 layers"));
        let mut bad = exported.clone();
        bad[1].layer_index = 2;
        assert!(check(hash, &bad)
            .unwrap_err()
            .to_string()
            .contains("geometry"));
    }

    /// A held snapshot builds a store holding exactly it — byte for byte through
    /// `export` — marked seeded so its first wave keeps it; and a snapshot that
    /// does not fit the geometry builds nothing.
    #[test]
    fn a_staged_snapshot_builds_a_seeded_store_holding_it() {
        let source = filled_store();
        let staged = StagedSnapshot {
            schedule_hash: source.schedule_hash(),
            layers: source.export().unwrap().into(),
        };
        let built = staged.build(&kinds(), &dims(), &Device::Cpu).unwrap();
        assert_eq!(built.export().unwrap(), source.export().unwrap());
        assert!(built.is_seeded());

        let foreign = StagedSnapshot {
            schedule_hash: staged.schedule_hash ^ 1,
            layers: staged.layers.clone(),
        };
        assert!(foreign.build(&kinds(), &dims(), &Device::Cpu).is_err());
    }

    /// **The last holder of a snapshot's rows returns them to the host pool**;
    /// a holder that shares them returns nothing, and the rows a decode then
    /// fills are the returned ones.
    #[test]
    fn a_dropped_snapshot_returns_its_rows_to_the_pool() {
        let _g = host_rows::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        host_rows::clear();
        let source = filled_store();
        let layers: Arc<[ExportedLayerState]> = source.export().unwrap().into();
        let bytes: usize = layers
            .iter()
            .map(|l| l.state.capacity() + l.conv_tail.capacity())
            .sum();
        let first_state = layers[0].state.as_ptr();
        let staged = StagedSnapshot {
            schedule_hash: source.schedule_hash(),
            layers: Arc::clone(&layers),
        };
        let shared = StagedSnapshot {
            schedule_hash: source.schedule_hash(),
            layers,
        };
        drop(staged);
        assert_eq!(host_rows::pooled_bytes(), 0, "the rows are still shared");
        let want = shared.layers[0].state.clone();
        drop(shared);
        assert_eq!(host_rows::pooled_bytes(), bytes);
        let refilled = (0..64)
            .map(|_| host_rows::row_from(&want))
            .find(|r| r.as_ptr() == first_state);
        assert!(refilled.is_some(), "a decode fills a returned row");
        host_rows::clear();
    }

    /// A fork carries the parent's state exactly. Byte equality through
    /// `export`, not a tolerance: this is a memory copy, and a tolerance would
    /// hide a layout bug behind "close enough".
    #[test]
    fn fork_carries_the_parents_state_exactly() {
        let parent = filled_store();
        let child = parent.fork_from().unwrap();
        assert_eq!(
            child.export().unwrap(),
            parent.export().unwrap(),
            "the fork must carry the parent's state byte for byte"
        );
        assert_eq!(child.schedule_hash(), parent.schedule_hash());
        assert_eq!(child.n_recurrent_layers(), parent.n_recurrent_layers());
        assert!(!child.holds_scratch());
    }

    /// **The `Clone`-shares-storage hazard.** `Tensor::clone` is a shallow handle
    /// clone, so a fork built from clones would look right and then track every
    /// mutation of its parent — or move its parent with every mutation of its own.
    #[test]
    fn fork_buffers_are_distinct_allocations() {
        let parent = filled_store();
        let mut child = parent.fork_from().unwrap();
        let parent_before = parent.export().unwrap();
        {
            let live = child.layer_state_mut(0).unwrap();
            let ones = Tensor::ones(live.s.shape(), live.s.dtype(), &Device::Cpu).unwrap();
            live.s.add_mut(&ones).unwrap();
            live.conv_tail
                .add_mut(
                    &Tensor::ones(live.conv_tail.shape(), live.conv_tail.dtype(), &Device::Cpu)
                        .unwrap(),
                )
                .unwrap();
        }
        assert_eq!(
            parent.export().unwrap(),
            parent_before,
            "the child shares the parent's live buffer"
        );
    }

    /// A fork mid-wave is refused: the wave's advance is not yet in the live
    /// state.
    #[test]
    fn fork_mid_wave_is_refused() {
        let mut store = filled_store();
        store.begin_wave().unwrap();
        let err = match store.fork_from() {
            Ok(_) => panic!("a mid-wave fork must be refused, not silently stale"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("mid-wave"), "{err}");
        store.commit_wave();
        assert!(store.fork_from().is_ok(), "a wave boundary is fine");
    }

    /// The fork reads the committed state: a long span exchanged in, or a short
    /// span folded, is in the child.
    #[test]
    fn fork_reads_the_committed_state() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        let before = store.export().unwrap();
        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        store.commit_wave();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 1.0, 1.0);
        store.commit_wave();
        store.fold(None).unwrap();
        let after = store.export().unwrap();
        assert_ne!(after[0], before[0]);
        assert_ne!(after[1], before[1]);
        let child = store.fork_from().unwrap();
        assert_eq!(child.export().unwrap(), after);
    }

    /// A fork is seeded: its state was put there deliberately, so the first
    /// `offset == 0` wave must not reset it — once.
    #[test]
    fn a_fork_is_seeded_exactly_once() {
        let parent = filled_store();
        let mut child = parent.fork_from().unwrap();
        assert!(child.is_seeded(), "a fresh fork carries seeded state");
        assert!(child.take_seeded(), "the first read reports it");
        assert!(
            !child.take_seeded(),
            "and consumes it — a second offset-0 wave resets normally"
        );
        assert!(
            !RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu)
                .unwrap()
                .is_seeded(),
            "a fresh store holds the sequence-start value already"
        );
    }

    /// Import is a restore, so it seeds for the same reason a fork does.
    #[test]
    fn import_seeds_the_store() {
        let store = filled_store();
        let exported = store.export().unwrap();
        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        assert!(!fresh.is_seeded());
        fresh.import(store.schedule_hash(), &exported).unwrap();
        assert!(
            fresh.is_seeded(),
            "a restored slot standing at offset 0 before its first wave looks \
             exactly like a fresh one — without the flag the reset wipes it"
        );
    }

    /// A refused import must not seed either: the store still holds zeros, and
    /// claiming otherwise would suppress the one reset that keeps it honest.
    #[test]
    fn a_refused_import_does_not_seed() {
        let store = filled_store();
        let exported = store.export().unwrap();
        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        assert!(fresh.import(store.schedule_hash() ^ 1, &exported).is_err());
        assert!(!fresh.is_seeded(), "a rejected restore seeded the store");
    }

    /// **The resume oracle.** Seal → drop the store entirely → resume from the
    /// exported rows → the state is bit-identical.
    #[test]
    fn seal_drop_resume_restores_a_bit_identical_state() {
        let sealed = {
            let store = filled_store();
            (store.schedule_hash(), store.export().unwrap())
        }; // the store is dropped here — nothing of it survives but the bytes

        let mut resumed = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        resumed.import(sealed.0, &sealed.1).unwrap();
        assert_eq!(
            resumed.export().unwrap(),
            sealed.1,
            "the resumed state must be byte-identical to the sealed one"
        );
    }

    /// A resume under a different model or a changed layer schedule refuses and
    /// leaves the store untouched, so the caller recomputes from a known state
    /// rather than from a half-scattered foreign one.
    #[test]
    fn resume_under_a_foreign_schedule_refuses_and_changes_nothing() {
        let store = filled_store();
        let exported = store.export().unwrap();

        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        let zeros = fresh.export().unwrap();
        assert!(fresh.import(store.schedule_hash() ^ 1, &exported).is_err());
        assert_eq!(fresh.export().unwrap(), zeros, "the refusal is total");
        assert!(!fresh.is_seeded(), "and it does not claim to be seeded");
    }

    /// Resume, then fork: a restored conversation forks exactly like a live
    /// one.
    #[test]
    fn a_resumed_store_forks_like_a_live_one() {
        let sealed = {
            let store = filled_store();
            (store.schedule_hash(), store.export().unwrap())
        };
        let mut resumed = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        resumed.import(sealed.0, &sealed.1).unwrap();

        let child = resumed.fork_from().unwrap();
        assert_eq!(child.export().unwrap(), sealed.1);
        assert!(child.is_seeded());
    }

    #[test]
    fn schedule_hash_pins_layout() {
        let h = schedule_hash(&kinds(), &dims());
        assert_eq!(h, schedule_hash(&kinds(), &dims()), "deterministic");
        let mut other = kinds();
        other[2] = LayerKind::DeltaNet;
        assert_ne!(h, schedule_hash(&other, &dims()), "schedule change");
        let mut d2 = dims();
        d2.conv_kernel = 4;
        assert_ne!(h, schedule_hash(&kinds(), &d2), "dims change");
    }
}
