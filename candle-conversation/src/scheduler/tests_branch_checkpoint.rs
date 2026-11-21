//! Tests for branch checkpoints and recurrent-state round trips: the state a
//! prefill produces is exported, stored, and installed without change.
//!
//! Byte equality is asserted instead of a tolerance, since these are copies and
//! any difference is a layout bug. A checkpoint from another model's schedule and
//! an aux blob from another model are both refused rather than read as a plausible
//! state. A snapshot that describes a turn the timeline did not recover is
//! rejected, and a fork or a release carries or drops both classes of state. Each
//! test drives the real handlers over a CPU double whose toy recurrence has exact
//! values, so a lost or duplicated turn shows up as a wrong number.

use super::*;

// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
// Scheduler unit tests
// •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
//
// The `DummyModel` is a minimal `ManagedBatchedModel` that returns dummy CPU
// tensors so no GPU is needed. `BatchedInferenceSession` is constructed on
// `Device::Cpu` with tiny arena dimensions so these tests run without CUDA.

#[cfg(test)]
pub(super) mod tests {
    // Expected token totals are written as the sum of the per-item terms they
    // came from, including the zero ones, so the arithmetic under test is
    // visible in the assertion.
    #![allow(clippy::identity_op)]

    use super::*;

    use super::super::tests_fixtures::tests::{make_test_scheduler, make_test_scheduler_recurrent};

    use super::super::tests_toy_model::tests::{
        DummyRecurrentModel, Pushed, ToyIndex, TOY_INDEX_RATIO, TOY_PAGE,
    };

    use super::super::tests_forks::tests::fork_log_scheduler;

    /// **A branch checkpoint is staged on its slots, not restored into them.**
    /// Restored at install, every new conversation held a whole recurrent store
    /// before its turn was admitted, and a window of them took the span from
    /// under a turn admission had already priced — the lost calibration cases.
    #[test]
    fn a_branch_checkpoint_install_stages_and_claims_nothing() {
        let (mut sched, log) = fork_log_scheduler(false);
        let slots: Vec<SequenceId> = (0..3)
            .map(|_| SequenceId(sched.session.create_sequence().unwrap()))
            .collect();
        let layer = ExportedLayerState {
            layer_index: 0,
            n_v_heads: 1,
            d_v: 1,
            d_k: 4,
            state: vec![0; 16],
            conv_channels: 0,
            conv_tail_cols: 0,
            conv_tail: Vec::new(),
        };
        let (tx, rx) = flume::bounded(1);
        sched.handle_request(SchedulerRequest::InstallRecurrentState {
            sequence_ids: slots.clone(),
            state: SharedState {
                schedule_hash: 1,
                layers: vec![layer].into(),
                aux: Vec::new().into(),
            },
            response_tx: tx,
        });

        assert!(rx.recv().unwrap().expect("staged, not restored"));
        let staged: Vec<usize> = log.staged.lock().unwrap().clone();
        assert_eq!(staged, slots.iter().map(|s| s.0).collect::<Vec<_>>());
        assert!(!slots.iter().any(|s| sched.model.has_recurrent_state(s.0)));
    }

    /// **A prefix section whose page the model will not take still advances
    /// the index, as a gap of the section's width.** Pages in order where the
    /// model reads them; the section with a stale page and the section with
    /// none each become a gap of exactly their tokens, so every later piece
    /// sits where its K/V does. Before, both only warned, left the index short
    /// of the slot, and every select of the ingest refused for as long as the
    /// daemon ran.
    #[test]
    fn a_prefix_section_the_model_cannot_index_is_advanced_over_as_a_gap() {
        let (mut sched, _tx, probe) = make_test_scheduler_recurrent();
        let (good, stale, missing) = (SectionId::new(1), SectionId::new(2), SectionId::new(3));
        // The slot's substrate registers each section against its stream, which
        // is what the pages are keyed by.
        let conversation = Conversation::new();
        for (section, stream) in [(good, 11), (stale, 12), (missing, 13)] {
            conversation.update(|substrate| {
                substrate.restore_section(
                    section,
                    StreamId(stream),
                    0,
                    Vec::new(),
                    Vec::new(),
                    Arc::new(Vec::new()),
                )
            });
        }
        let seq = SequenceId(5);
        sched.slot_conversations.insert(seq, conversation.clone());
        let good_page = [TOY_PAGE, b"-good"].concat();
        sched.section_positional.insert(
            StreamId(11),
            &conversation,
            Arc::new(StoredPages::new(good_page.clone())),
        );
        sched.section_positional.insert(
            StreamId(12),
            &conversation,
            Arc::new(StoredPages::new(b"\x02\x00\x00\x00rotated".to_vec())),
        );

        for (section, tokens) in [(good, 40), (stale, 23), (missing, 17)] {
            let stream = conversation.read().section_stream(section);
            sched.push_prefix_section_index(seq, section, stream, tokens);
        }

        assert_eq!(
            probe
                .pushed
                .lock()
                .unwrap()
                .get(&5)
                .cloned()
                .unwrap_or_default(),
            vec![Pushed::Page(good_page), Pushed::Gap(23), Pushed::Gap(17)],
        );
    }

    // ── Branch checkpoints (P6) ──────────────────────────────────────────────
    //
    // These run the real handlers on the CPU double. Before it reported
    // `carries_recurrent_state`, `handle_branch_checkpoint_pass` returned
    // `Ok(None)` on its first line and none of this path was reachable below
    // tier 3 — which is where the write/read race below was actually caught, at
    // ten minutes and 22 GB a run.

    /// **The pass prefills the tokens and hands back the state they produced.**
    ///
    /// Not a copy of anything: the checkpoint's whole reason to exist is that a
    /// recurrence cannot be Arc-injected, so the tokens must genuinely pass
    /// through the model. A pass that returned the sequence-start state would
    /// look identical from the caller's side.
    #[test]
    fn the_branch_checkpoint_pass_returns_state_the_tokens_produced() {
        let (mut sched, _tx, _probe) = make_test_scheduler_recurrent();
        let state = sched
            .handle_branch_checkpoint_pass(None, &[1, 2, 3, 4])
            .expect("pass ran")
            .expect("a model that carries state returns some");
        assert_eq!(state.schedule_hash, DummyRecurrentModel::SCHEDULE_HASH);
        assert_eq!(state.layers.len(), 2, "one row per toy layer");
        let advanced = state.layers.iter().any(|l| l.state.iter().any(|&b| b != 0));
        assert!(
            advanced,
            "the exported state is all zeros — the tokens never reached the model, \
             and every conversation restoring this would start with no memory of \
             its system prompt while reading perfectly"
        );
    }

    /// **The pass leaves no slot behind.**
    ///
    /// It allocates a throwaway sequence per call and runs once per new branch
    /// per process. A slot leaked here is a slot leaked for the life of the
    /// daemon.
    #[test]
    fn the_branch_checkpoint_pass_frees_its_slot() {
        let (mut sched, _tx, probe) = make_test_scheduler_recurrent();
        let before = probe.len();
        for _ in 0..4 {
            sched
                .handle_branch_checkpoint_pass(None, &[7, 7, 7])
                .expect("pass ran");
        }
        assert_eq!(
            probe.len(),
            before,
            "four passes left {} live state entries — the throwaway slots are not \
             being released",
            probe.len() - before
        );
    }

    /// **A model carrying no recurrent state pays nothing.**
    ///
    /// The gate that keeps this whole path off a plain transformer. `DummyModel`
    /// answers `false`, so the pass must return before allocating anything.
    #[test]
    fn the_branch_checkpoint_pass_is_a_no_op_without_recurrent_state() {
        let (mut sched, _tx) = make_test_scheduler();
        assert!(
            sched
                .handle_branch_checkpoint_pass(None, &[1, 2, 3])
                .expect("pass ran")
                .is_none(),
            "a model with no recurrent state must not run a prefill for a \
             checkpoint nothing will read"
        );
    }

    /// **A checkpoint round-trips through its record and installs.**
    ///
    /// The whole P6 loop below the conversation layer: export → encode → decode
    /// → install, with the state byte-identical at the far end. Byte equality,
    /// not tolerance — these are copies, so any difference is a layout bug.
    #[test]
    fn a_branch_checkpoint_round_trips_through_its_record_and_installs() {
        use crate::persistence::content_hash::ContentHash;
        use crate::persistence::record::{BranchCheckpointPayload, SnapshotLayer};

        let (mut sched, _tx, probe) = make_test_scheduler_recurrent();
        let state = sched
            .handle_branch_checkpoint_pass(None, &[5, 6, 7, 8])
            .expect("pass ran")
            .expect("state");
        let computed = state.layers.clone();

        let payload = BranchCheckpointPayload {
            prefix_hash: ContentHash { lo: 9, hi: 11 },
            schedule_hash: state.schedule_hash,
            layers: state.layers.into_iter().map(SnapshotLayer::from).collect(),
            aux: state.aux,
        };
        let back = BranchCheckpointPayload::decode(&payload.encode()).expect("decode");
        assert_eq!(back, payload, "the record did not survive its own encoding");

        // Install onto a fresh slot and read the state back out.
        let slot = sched.session.create_sequence().expect("slot");
        let restored: Vec<ExportedLayerState> = back
            .layers
            .into_iter()
            .map(ExportedLayerState::from)
            .collect();
        assert!(sched
            .model
            .restore_recurrent(slot, back.schedule_hash, &restored)
            .expect("restore"));

        let (_, after) = sched
            .model
            .export_recurrent(slot)
            .expect("export")
            .expect("state");
        assert_eq!(
            after.iter().map(|l| l.state.clone()).collect::<Vec<_>>(),
            computed.iter().map(|l| l.state.clone()).collect::<Vec<_>>(),
            "the installed state is not the computed one"
        );
        assert!(
            probe.is_seeded(slot),
            "a restored slot must be seeded, or its first wave resets the state it \
             was just given back to zero — defect 6 wearing the fix's clothes"
        );
    }

    /// **A checkpoint from another model's schedule is refused, not scattered.**
    #[test]
    fn a_branch_checkpoint_under_a_foreign_schedule_is_refused() {
        let (mut sched, _tx, _probe) = make_test_scheduler_recurrent();
        let state = sched
            .handle_branch_checkpoint_pass(None, &[3, 1, 4])
            .expect("pass ran")
            .expect("state");
        let slot = sched.session.create_sequence().expect("slot");
        let err = sched
            .model
            .restore_recurrent(slot, 0xBAD, &state.layers)
            .expect_err("a foreign schedule hash must not install");
        assert!(err.to_string().contains("schedule hash"), "{err}");
    }

    /// **`InstallRecurrentState` is the request the conversation layer uses.**
    ///
    /// Dispatched through `handle_request`, so the wiring is covered and not
    /// just the handler body.
    #[test]
    fn install_recurrent_state_request_installs_onto_the_named_slot() {
        let (mut sched, _tx, probe) = make_test_scheduler_recurrent();
        let state = sched
            .handle_branch_checkpoint_pass(None, &[2, 4, 6])
            .expect("pass ran")
            .expect("state");
        let slot = SequenceId(sched.session.create_sequence().expect("slot"));

        let (tx, rx) = flume::bounded(1);
        sched.handle_request(SchedulerRequest::InstallRecurrentState {
            sequence_ids: vec![slot],
            state: SharedState::from(&state),
            response_tx: tx,
        });
        assert!(
            rx.recv().expect("reply").expect("install"),
            "the install reported that no state was carried"
        );
        assert!(probe.get(slot.0).is_some(), "the slot has no state");
    }

    /// **A pass continued from a prefix's state runs on from it** — what a
    /// chain of sections relies on: an entry sealed from the state the entries
    /// before it leave is computed after them, not from zero. The toy
    /// recurrence advances per forward (`s ← 2s + t·k`), so four tokens and
    /// then five land on `2·4k + 5k = 13k`, and five from zero on `5k`. Exact
    /// in f32, so compared without tolerance. Both recurrences carry over: the
    /// index stands at all nine tokens.
    #[test]
    fn a_pass_continued_from_a_state_runs_on_from_it() {
        let (mut sched, _tx, _probe) = make_test_scheduler_recurrent();
        let head = sched
            .handle_branch_checkpoint_pass(None, &[1, 2, 3, 4])
            .expect("pass ran")
            .expect("state");
        let continued = sched
            .handle_branch_checkpoint_pass(Some(&SharedState::from(&head)), &[5, 6, 7, 8, 9])
            .expect("pass ran")
            .expect("state");
        let cells = |s: &ExportedState| -> Vec<f32> {
            s.layers
                .iter()
                .flat_map(|l| l.state.chunks_exact(4))
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let expected: Vec<f32> = (1..=8).map(|k| 13.0 * k as f32).collect();
        assert_eq!(
            cells(&continued),
            expected,
            "the continuation lost the prefix's state"
        );
        let index = ToyIndex::decode(&continued.aux).expect("the aux blob is this model's");
        assert_eq!(index.tokens, 9, "the index did not carry over");
    }

    /// **A section seal starts from the state it is given.** Staged before the
    /// slot claims its carried state, so the claim takes it as the slot's store
    /// rather than building one from zero — the state a prefill then runs on.
    #[test]
    fn a_section_seal_starts_from_the_state_it_is_given() {
        let (mut sched, _tx, probe) = make_test_scheduler_recurrent();
        let start = sched
            .handle_branch_checkpoint_pass(None, &[2, 4, 6])
            .expect("pass ran")
            .expect("state");
        let slot = SequenceId(sched.session.create_sequence().expect("slot"));
        let (response_tx, _rx) = flume::bounded(1);
        let queued = QueuedSection {
            sequence_id: slot,
            section_id: SectionId::new(1),
            prefix_section_ids: Vec::new(),
            tokens: TokenBuffer::from(vec![1u32; 4]),
            address: ContentAddress::default(),
            debug_name: "s".into(),
            in_collection: true,
            start: Some(SharedState::from(&start)),
            band: DecodePriority::Low,
            response_tx,
            queued_at: Instant::now(),
        };
        assert!(
            sched.begin_section(queued).is_ok(),
            "the section was refused"
        );
        assert!(
            sched.active_section_ingests[0].export_end,
            "a section given its start must hand back its end, the start of what continues it"
        );
        let (_, held) = sched
            .model
            .export_recurrent(slot.0)
            .expect("export")
            .expect("the slot holds a state");
        assert_eq!(
            held.iter().map(|l| l.state.clone()).collect::<Vec<_>>(),
            start
                .layers
                .iter()
                .map(|l| l.state.clone())
                .collect::<Vec<_>>(),
            "the slot's state before its prefill is not the one it was given"
        );
        assert!(probe.get(slot.0).is_some(), "the slot has no state");
    }

    // —— Store and resume: both classes of state ————————————————————————————
    //
    // Everything below is about the SECOND class — the model's own encoding,
    // carried as opaque bytes beside the delta-rule layers. The layers have
    // their own coverage above; these exist because the two classes travel the
    // same paths and only one of them was ever asserted, which is exactly the
    // shape of a state that goes missing without a failing test.

    /// **The pass exports BOTH classes, and they agree about the tokens.**
    ///
    /// The failure this rules out is a pass that returns the layers a prefill
    /// produced beside an aux blob it did not — which reads as a complete
    /// checkpoint and installs a sequence whose two halves of memory stand at
    /// different depths.
    #[test]
    fn the_branch_checkpoint_pass_exports_both_classes_of_state() {
        let (mut sched, _tx, _probe) = make_test_scheduler_recurrent();
        let tokens = [1u32, 2, 3, 4, 5, 6, 7];
        let state = sched
            .handle_branch_checkpoint_pass(None, &tokens)
            .expect("pass ran")
            .expect("state");
        assert!(!state.layers.is_empty(), "no delta-rule layers");
        let idx = ToyIndex::decode(&state.aux).expect("the aux blob is this model's");
        assert_eq!(
            idx.tokens,
            tokens.len() as u32,
            "the index stands at a different depth than the prefill that made it"
        );
        assert_eq!(idx.rows(), 1, "7 tokens at ratio 4 is one whole row");
        assert_eq!(
            idx.open(),
            3,
            "…and a ragged open block of 3 — the case a boundary-aligned \
             fixture would never exercise"
        );
    }

    /// **A ragged boundary survives the record verbatim.**
    ///
    /// A turn ends where the text ends, not on a block boundary, so the open
    /// block is the normal case rather than an edge one. Every remainder is
    /// checked because a reconstruction that rounded to whole rows would still
    /// pass at remainder 0.
    #[test]
    fn every_ragged_boundary_round_trips_through_the_snapshot_record() {
        use crate::persistence::record::{SnapshotLayer, SnapshotPayload};

        for extra in 0..TOY_INDEX_RATIO {
            let (mut sched, _tx, _probe) = make_test_scheduler_recurrent();
            let n = (2 * TOY_INDEX_RATIO + extra) as usize;
            let tokens: Vec<u32> = (0..n as u32).collect();
            let state = sched
                .handle_branch_checkpoint_pass(None, &tokens)
                .expect("pass ran")
                .expect("state");

            let payload = SnapshotPayload {
                timeline_id: 3,
                turn_index: 0,
                schedule_hash: state.schedule_hash,
                layers: state.layers.into_iter().map(SnapshotLayer::from).collect(),
                aux: state.aux.clone(),
            };
            let back = SnapshotPayload::decode(&payload.encode()).expect("decode");
            assert_eq!(back.aux, state.aux, "the record altered the blob");

            let idx = ToyIndex::decode(&back.aux).expect("decode index");
            assert_eq!(idx.tokens, n as u32);
            assert_eq!(idx.rows(), 2 + extra / TOY_INDEX_RATIO);
            assert_eq!(idx.open(), extra, "the partial row's width was lost");
        }
    }

    /// **Install puts the aux blob on the slot, and the value is the one that
    /// was exported.**
    ///
    /// Through `handle_request`, so the wiring is covered rather than the
    /// handler body alone.
    #[test]
    fn install_recurrent_state_request_installs_the_aux_blob_too() {
        let (mut sched, _tx, probe) = make_test_scheduler_recurrent();
        let state = sched
            .handle_branch_checkpoint_pass(None, &[2, 4, 6, 8, 10])
            .expect("pass ran")
            .expect("state");
        let slot = SequenceId(sched.session.create_sequence().expect("slot"));

        let (tx, rx) = flume::bounded(1);
        sched.handle_request(SchedulerRequest::InstallRecurrentState {
            sequence_ids: vec![slot],
            state: SharedState::from(&state),
            response_tx: tx,
        });
        assert!(rx.recv().expect("reply").expect("install"));
        assert_eq!(
            probe.get_index(slot.0).expect("the slot has no index"),
            ToyIndex { tokens: 5 },
            "the slot's index is not the one the checkpoint carried"
        );
    }

    /// **A restored slot is seeded, so its first wave does not reset the index
    /// it was just given.**
    ///
    /// The same trap the delta-rule restore has: a restored sequence stands at
    /// offset 0, and `ensure` resets an occupied slot at offset 0 unless it was
    /// seeded. An aux restore that installed the bytes without seeding would put
    /// the index back to zero on the next forward — and nothing downstream would
    /// report it.
    #[test]
    fn an_aux_restore_survives_the_first_waves_offset_zero_reset() {
        let (mut sched, _tx, probe) = make_test_scheduler_recurrent();
        let slot = sched.session.create_sequence().expect("slot");
        sched
            .model
            .restore_aux_state(slot, &ToyIndex { tokens: 9 }.encode())
            .expect("restore");

        // The reset rule is `ensure`'s, and `ensure` is what a wave runs first.
        probe.ensure(slot, 0);
        assert_eq!(
            probe.get_index(slot).expect("index gone entirely"),
            ToyIndex { tokens: 9 },
            "the restored index did not survive the offset-0 wave — the slot \
             was not seeded, and the conversation resumes having forgotten"
        );
    }

    /// **A blob from another model is refused, not read as a plausible state.**
    ///
    /// The aux blob is opaque to everything between the model and the record,
    /// so the model is the only place this can be caught. Reading a foreign
    /// blob as a state is worse than failing to read it: the conversation then
    /// resumes at a depth nothing computed.
    #[test]
    fn a_foreign_aux_blob_is_refused() {
        let (mut sched, _tx, probe) = make_test_scheduler_recurrent();
        let slot = sched.session.create_sequence().expect("slot");
        for bad in [
            b"XIDX\x01\x00\x00\x00".to_vec(), // right shape, wrong magic
            b"TIDX\x01\x00\x00".to_vec(),     // right magic, truncated
            Vec::new(),
        ] {
            assert!(
                sched.model.restore_aux_state(slot, &bad).is_err(),
                "a {}-byte foreign blob installed as a state",
                bad.len()
            );
        }
        assert!(
            probe.get_index(slot).is_none(),
            "a refused restore still left something on the slot"
        );
    }

    /// **A fork copies both classes.**
    ///
    /// A view carve that copied the layers alone would give the child a
    /// delta-rule state at the parent's depth and an index at zero.
    #[test]
    fn a_fork_carries_both_classes_of_state() {
        let (mut sched, _tx, probe) = make_test_scheduler_recurrent();
        let parent = sched.session.create_sequence().expect("parent");
        let child = sched.session.create_sequence().expect("child");
        sched
            .model
            .restore_aux_state(parent, &ToyIndex { tokens: 13 }.encode())
            .expect("seed the parent");
        probe.set(parent, [[1.0; 4]; 2]);

        sched.model.fork_recurrent(parent, child).expect("fork");
        assert_eq!(probe.get(child), Some([[1.0; 4]; 2]), "layers not forked");
        assert_eq!(
            probe.get_index(child),
            Some(ToyIndex { tokens: 13 }),
            "the child's index is at a different depth than its layers"
        );
    }

    /// **Releasing a slot releases both classes.**
    ///
    /// The branch checkpoint pass allocates a throwaway slot per call. A class
    /// of state the release does not drop leaks once per new branch, for the
    /// life of the daemon — and the existing leak test only counts the layers.
    #[test]
    fn releasing_a_slot_drops_both_classes_of_state() {
        let (mut sched, _tx, probe) = make_test_scheduler_recurrent();
        let before = probe.index.lock().unwrap().len();
        for _ in 0..4 {
            sched
                .handle_branch_checkpoint_pass(None, &[7, 7, 7])
                .expect("pass ran");
        }
        assert_eq!(
            probe.index.lock().unwrap().len(),
            before,
            "four passes leaked {} index entries",
            probe.index.lock().unwrap().len() - before
        );
    }

    /// **One request installs the same branch state onto EVERY named slot.**
    ///
    /// This is what lets a batch of conversations born on one prompt branch pay
    /// a single queue wait instead of one per conversation, so the multi-slot
    /// case is asserted directly rather than inferred from the single-slot one.
    #[test]
    fn install_recurrent_state_request_installs_onto_every_named_slot() {
        let (mut sched, _tx, probe) = make_test_scheduler_recurrent();
        let state = sched
            .handle_branch_checkpoint_pass(None, &[2, 4, 6])
            .expect("pass ran")
            .expect("state");
        let slots: Vec<SequenceId> = (0..3)
            .map(|_| SequenceId(sched.session.create_sequence().expect("slot")))
            .collect();

        let (tx, rx) = flume::bounded(1);
        sched.handle_request(SchedulerRequest::InstallRecurrentState {
            sequence_ids: slots.clone(),
            state: SharedState::from(&state),
            response_tx: tx,
        });
        assert!(
            rx.recv().expect("reply").expect("install"),
            "the install reported that no state was carried"
        );
        for slot in &slots {
            assert!(
                probe.get(slot.0).is_some(),
                "slot {slot} was named in the request but has no state"
            );
        }
    }

    /// **T7.3 — a snapshot newer than the recovered history is rejected.**
    ///
    /// The seal writes the snapshot *before* the turn's `Tokens` record, on the
    /// stated grounds that the resulting tear — a snapshot for a turn whose
    /// records never landed — is "handled". This is the handling. Without it the
    /// write ordering is actively harmful: it guarantees the tear it cannot
    /// survive, and a resume installs a state one turn ahead of its K/V.
    ///
    /// The boundary is the whole test. Turns are `0..n`, so a snapshot at `n` is
    /// the first invalid one, and an off-by-one here accepts exactly the torn
    /// case the ordering was chosen to produce.
    #[test]
    fn a_snapshot_newer_than_the_recovered_history_is_rejected() {
        // Three turns recovered: 0, 1, 2.
        assert!(snapshot_within_recovered_history(0, 3));
        assert!(snapshot_within_recovered_history(2, 3), "the last turn");
        assert!(
            !snapshot_within_recovered_history(3, 3),
            "turn 3 does not exist when 3 turns recovered — this is the torn \
             shutdown the seal ordering deliberately allows"
        );
        assert!(!snapshot_within_recovered_history(9, 3));

        // Nothing recovered: every snapshot is ahead of the history, including
        // the first. A conversation whose records were all lost must not inherit
        // a state describing them.
        assert!(!snapshot_within_recovered_history(0, 0));
    }
}
