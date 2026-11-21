//! Tests for slot teardown and carried belief: freed slots take their queued work
//! and their belief with them, and a shared substrate is walked once.
//!
//! A carried belief is keyed by its slot, so a reset or freed slot clears it and
//! one occupant's belief never seeds another's projections. A resumed slot seeds
//! its belief from the selections it recovered, folded in turn order, and a
//! timeline with none seeds nothing. Freeing a member of the wave cohort abandons
//! the cohort so that the recycled id cannot re-enter the next group, while
//! freeing a non-member leaves the survivors' progress intact. A substrate shared
//! by several slots is walked once, and an attached substrate with no slot is
//! still walked.

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

    use super::super::tests_fixtures::tests::{create_scratch_slot, make_test_scheduler};

    use super::super::tests_admission::tests::test_prefill_work;

    use super::super::tests_ephemeral_forks::tests::{park_fork, start_turn_on};

    // —— View swap tests ——————————————————————————————————————————————————————
    //
    // The previous `swap_view_with_new_ranges` mid-decode view-swap path
    // has been replaced by the zero-copy reproject rebuild in
    // [`Scheduler::reproject_view`].  Its dedicated unit test
    // (`swap_view_re_keys_state_and_preserves_turn_start_parent_blocks`)
    // has been removed alongside the function.  The same re-keying
    // invariants — `active_decodes` / `sampling_states` / `turn_views`
    // migrate to the new view id, `DecodeState.event_tx` survives —
    // are now exercised end-to-end by the `zend` coherence integration
    // test, which fires reproject many times in a real decode loop.

    // —— Carried-belief slot lifecycle ————————————————————————————————————————
    //
    // The belief carried across a conversation's turns is keyed by its parent
    // slot; slot teardown and reuse must never let one occupant's belief seed
    // another's projections.

    /// A carried belief for a slot.
    pub(in super::super) fn seeded_belief() -> PriorBelief {
        let mut b = PriorBelief::default();
        b.set("tools", "calculator", 2500.0, true, true);
        b
    }

    #[test]
    fn reset_sequence_clears_the_slots_carried_belief() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let raw_id = scheduler.session.create_sequence().expect("create");
        let seq_id = SequenceId(raw_id);
        scheduler.carried_beliefs.insert(seq_id, seeded_belief());

        let (rtx, rrx) = flume::bounded(1);
        scheduler.handle_request(SchedulerRequest::ResetSequence {
            sequence_id: seq_id,
            response_tx: rtx,
        });
        rrx.recv().expect("reset response").expect("reset ok");

        assert!(
            !scheduler.carried_beliefs.contains_key(&seq_id),
            "a reset slot is reused for new content — the previous occupant's \
             belief must not survive the reset"
        );
    }

    /// Carve a turn view over `parent` and register it the way SubmitTurn does.
    pub(in super::super) fn register_turn_view(
        scheduler: &mut Scheduler,
        parent: SequenceId,
    ) -> SequenceId {
        let (view, borrowed) = scheduler.create_view(parent, &[]).expect("carve");
        scheduler.turn_views.insert(
            view,
            ViewState {
                parent_id: parent,
                original_borrowed: borrowed,
                turn_start_parent_blocks: borrowed.0,
                question_tokens: 0,
                applied_identity: None,
                state_forked: true,
                holds_parent_state: false,
            },
        );
        view
    }

    /// **A turn view cannot outlive the parent it borrows from.** Freeing the
    /// parent used to leave the view registered and its slot live, holding the
    /// parent's prefix borrowed with nothing left to release it.
    #[test]
    fn freeing_a_parent_frees_its_turn_view() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let parent = SequenceId(scheduler.session.create_sequence().expect("create"));
        let view = register_turn_view(&mut scheduler, parent);

        scheduler.handle_request(SchedulerRequest::FreeSequence {
            sequence_id: parent,
        });

        assert!(scheduler.turn_views.is_empty(), "the view is unregistered");
        assert!(
            scheduler.session.sequence_offset(view.0).is_none(),
            "the view's slot is released"
        );
    }

    /// A reset discards the prefix a turn view borrows, so it takes the view.
    #[test]
    fn resetting_a_parent_frees_its_turn_view() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let parent = SequenceId(scheduler.session.create_sequence().expect("create"));
        let view = register_turn_view(&mut scheduler, parent);

        let (rtx, rrx) = flume::bounded(1);
        scheduler.handle_request(SchedulerRequest::ResetSequence {
            sequence_id: parent,
            response_tx: rtx,
        });
        rrx.recv().expect("reset response").expect("reset ok");

        assert!(scheduler.turn_views.is_empty(), "the view is unregistered");
        assert!(
            scheduler.session.sequence_offset(view.0).is_none(),
            "the view's slot is released"
        );
        assert!(
            scheduler.session.sequence_offset(parent.0).is_some(),
            "the reset parent stays live"
        );
    }

    /// **A resumed slot seeds its carried belief from the recovered selection
    /// events — the same fold the live seal harvest performs.**
    ///
    /// `carried_beliefs` is what a submit-time projection seeds from, and it is
    /// keyed by slot, so before this a resumed conversation selected its
    /// sections and tools from an EMPTY prior while its uninterrupted twin
    /// carried a prior evolved over every turn. Near a selection gate that
    /// flips a section in or out of the context, and the first resumed decode
    /// diverges — deterministically, marginally, and past every recall probe.
    /// A1 of the behavioural catalogue caught it; this pins the fix at unit
    /// scope so it never again needs two model loads to check.
    #[test]
    fn a_resumed_slot_seeds_carried_belief_from_the_recovered_selections() {
        use crate::persistence::content_hash::turn_stream_id;
        use crate::projection::event::{ProjectionEvent, ProjectionSelection, SystemItem};
        use crate::projection::SelectedSection;
        use crate::substrate::TurnPartWrite;

        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let timeline = TimelineId::for_test(11);
        conversation.register_timeline(
            timeline,
            crate::projection::LayerId::from_raw(1).unwrap(),
            crate::projection::GroupId::from_raw(1).unwrap(),
        );

        // Two recovered turns whose final projection events both name the same
        // tools-catalog member: the merge must fold them in TURN ORDER, so the
        // later turn's belief wins — exactly what the live harvest holds.
        for (expected_idx, score) in [(0u32, 0.25f32), (1u32, 0.75f32)] {
            let idx = conversation
                .record_turn(
                    timeline,
                    Role::Assistant,
                    TurnPartWrite {
                        token_count: 1,
                        block_end: 1,
                        sealed_gpu: Some(Arc::new(vec![])),
                        ..Default::default()
                    },
                    |_| Ok(vec![]),
                )
                .expect("record turn");
            assert_eq!(idx.0, expected_idx, "turns land in order");
            let event = ProjectionEvent {
                selection: ProjectionSelection {
                    system: vec![SystemItem::Collection {
                        name: "tools".into(),
                        member_glue: String::new(),
                        member_glue_tokens: 0,
                        sections: vec![SelectedSection {
                            name: "calculator".into(),
                            tokens: 5,
                            selected: true,
                            score,
                            qualified: true,
                        }],
                    }],
                    turns: vec![],
                },
                ..Default::default()
            };
            conversation.persist_projection_events(
                turn_stream_id(timeline.raw(), idx.0),
                encode_events(&[event]),
            );
        }

        let slot = SequenceId(scheduler.session.create_sequence().expect("slot"));
        scheduler.restore_carried_belief(slot, &conversation, timeline);

        let belief = scheduler
            .carried_beliefs
            .get(&slot)
            .expect("a resumed slot must not select from an empty prior");
        assert_eq!(
            belief.coll("tools").get("calculator"),
            Some(&(0.75, true, true)),
            "the recovered belief must be the turn-ordered fold of the persisted \
             selections — the last turn's stamped score, not the first's"
        );

        // A timeline with no recovered events seeds nothing: an empty prior IS
        // right for a genuinely fresh conversation, and inserting a default
        // would erase that distinction.
        let fresh = SequenceId(scheduler.session.create_sequence().expect("slot"));
        scheduler.restore_carried_belief(fresh, &conversation, TimelineId::for_test(12));
        assert!(
            !scheduler.carried_beliefs.contains_key(&fresh),
            "no recovered selections must mean no seeded belief"
        );
    }

    #[test]
    fn free_sequence_clears_the_slots_carried_belief() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let raw_id = scheduler.session.create_sequence().expect("create");
        let seq_id = SequenceId(raw_id);
        scheduler.carried_beliefs.insert(seq_id, seeded_belief());

        scheduler.handle_request(SchedulerRequest::FreeSequence {
            sequence_id: seq_id,
        });

        assert!(
            !scheduler.carried_beliefs.contains_key(&seq_id),
            "a freed slot id is recycled by the allocator — the dead \
             conversation's belief must not seed the next occupant"
        );
    }

    #[test]
    fn free_sequence_resets_wave_cohort_when_freed_slot_was_a_member() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let seq_id = SequenceId(scheduler.session.create_sequence().expect("create"));
        // The freed slot is a live member of the in-flight continuous-fair-wave
        // cohort (the code_read fork-churn case). Its held residual is indexed by
        // member position, so the whole cohort must be abandoned on free —
        // otherwise the recycled slot id re-enters the next wave-group assembly
        // alongside the stale member and trips the "appears in more than one
        // group" guard.
        scheduler.wave_prefill_members = vec![WaveMember::Prefill {
            seq_id: seq_id.0,
            advance: 1,
        }];
        scheduler.wave_prefill_cursor = 7;

        scheduler.handle_request(SchedulerRequest::FreeSequence {
            sequence_id: seq_id,
        });

        assert!(
            scheduler.wave_prefill_members.is_empty(),
            "freeing a held-cohort member must abandon the cohort so a recycled \
             slot id cannot duplicate it in the next wave-group assembly"
        );
        assert_eq!(
            scheduler.wave_prefill_cursor, 0,
            "abandoning the cohort must also clear the creep cursor"
        );
    }

    #[test]
    fn free_sequence_leaves_wave_cohort_intact_for_a_non_member() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let freed = SequenceId(scheduler.session.create_sequence().expect("create"));
        let member = SequenceId(scheduler.session.create_sequence().expect("create"));
        // A DIFFERENT slot is the sole cohort member. Freeing `freed` must leave
        // the held cohort untouched — resetting it on every unrelated free would
        // throw away the survivors' partial prefill progress each wave.
        scheduler.wave_prefill_members = vec![WaveMember::Prefill {
            seq_id: member.0,
            advance: 1,
        }];
        scheduler.wave_prefill_cursor = 3;

        scheduler.handle_request(SchedulerRequest::FreeSequence { sequence_id: freed });

        assert_eq!(
            scheduler.wave_prefill_members.len(),
            1,
            "freeing a non-member slot must leave the in-flight cohort intact"
        );
        assert_eq!(scheduler.wave_prefill_cursor, 3);
    }

    /// **A queued turn its caller let go of leaves before the next step**; a
    /// held one stays.
    #[test]
    fn a_let_go_prefill_is_withdrawn_and_a_held_one_stays() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let held = SequenceId(scheduler.session.create_sequence().expect("create"));
        let queued_let_go = SequenceId(scheduler.session.create_sequence().expect("create"));
        let dropped = |id| {
            let mut w = test_prefill_work(id);
            let (tx, rx) = flume::bounded(1);
            drop(rx);
            w.event_tx = tx;
            w
        };
        scheduler.prefill_queue.push_back(test_prefill_work(held));
        scheduler.prefill_queue.push_back(dropped(queued_let_go));
        scheduler.withdraw_abandoned_prefills();
        assert_eq!(
            scheduler
                .prefill_queue
                .iter()
                .map(|w| w.sequence_id)
                .collect::<Vec<_>>(),
            vec![held],
            "the queued let-go turn leaves, the held one stays"
        );
        assert!(scheduler.active_prefills.is_empty());
    }

    /// **A substrate shared by several slots is walked once.** Three slots on one
    /// workspace and one on another give two substrates, whichever slot registered
    /// them.
    #[test]
    fn slots_sharing_a_substrate_are_walked_once() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let workspace = crate::projection::Conversation::new();
        let other = crate::projection::Conversation::new();
        for _ in 0..3 {
            create_scratch_slot(&mut scheduler, &workspace);
        }
        create_scratch_slot(&mut scheduler, &other);

        let distinct = scheduler.distinct_substrates();
        assert_eq!(distinct.len(), 2);
        assert!(distinct.iter().any(|c| c.same_substrate(&workspace)));
        assert!(distinct.iter().any(|c| c.same_substrate(&other)));
        assert!(!workspace.same_substrate(&other));
    }

    /// **Detaching a substrate frees its slots and only its slots**, and drops
    /// its persistence trigger: the other tenant's slots and trigger are
    /// untouched, so its conversations go on being served.
    #[test]
    fn detaching_a_substrate_frees_only_its_slots() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let zend = crate::projection::Conversation::new();
        let npcd = crate::projection::Conversation::new();
        let kept: Vec<SequenceId> = (0..2)
            .map(|_| create_scratch_slot(&mut scheduler, &zend))
            .collect();
        for _ in 0..3 {
            create_scratch_slot(&mut scheduler, &npcd);
        }
        scheduler
            .persist_triggers
            .attach(1, PersistenceTrigger::noop());
        assert_eq!(scheduler.persist_triggers.len(), 2);

        scheduler.detach_substrate(1, &npcd);

        assert_eq!(scheduler.slot_conversations.len(), 2);
        for slot in &kept {
            assert!(scheduler.slot_conversations[slot].same_substrate(&zend));
        }
        assert_eq!(scheduler.distinct_substrates().len(), 1);
        assert_eq!(scheduler.persist_triggers.len(), 1);
    }

    /// **An attached substrate with no live slot is still walked.** Its residences
    /// hold resident KV, and a compaction sweep that skipped it relocated the chunks
    /// they name without rewriting them — on a shared engine, every pass run while
    /// one tenant was between conversations. Detaching it stops the walk; visiting
    /// it twice — attached and slotted — still walks it once.
    #[test]
    fn an_attached_substrate_with_no_slot_is_walked() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let zend = crate::projection::Conversation::new();
        let npcd = crate::projection::Conversation::new();
        scheduler.register_substrate(&zend);
        scheduler.register_substrate(&npcd);
        scheduler.register_substrate(&npcd);
        create_scratch_slot(&mut scheduler, &zend);

        let distinct = scheduler.distinct_substrates();
        assert_eq!(distinct.len(), 2, "npcd has no slot and is walked anyway");
        assert!(distinct.iter().any(|c| c.same_substrate(&npcd)));

        scheduler.detach_substrate(1, &npcd);
        let distinct = scheduler.distinct_substrates();
        assert_eq!(distinct.len(), 1);
        assert!(distinct[0].same_substrate(&zend));
    }

    /// **A requester that gave up costs no slot.**
    #[test]
    fn a_parked_fork_whose_requester_left_takes_no_slot() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let view = start_turn_on(&mut scheduler, parent);
        let rx = park_fork(&mut scheduler, parent);
        drop(rx);

        scheduler.turn_views.remove(&view);
        let slots = scheduler.slot_conversations.len();
        scheduler.drain_parked_forks();
        assert_eq!(scheduler.slot_conversations.len(), slots);
        assert!(scheduler.parked_forks.is_empty());
    }
}
