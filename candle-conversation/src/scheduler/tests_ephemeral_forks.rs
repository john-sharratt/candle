//! Tests for ephemeral forks: a question put beside a conversation forks its
//! parent's state, decodes at its asker's band, and never writes to the parent.
//!
//! A fork is refused while a turn runs on its parent, because the fork's view
//! would borrow the blocks the fork carves, and a request that arrives mid-turn
//! waits for the boundary. Requests parked on one parent each receive their own
//! fork, and a parent freed while a request waits fails that request, so the
//! caller does not hang. Freeing a fork leaves the parent's blocks and offset as
//! they were, since the fork writes only to chunks of its own.

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

    use candle::Tensor;

    use super::super::tests_fixtures::tests::{
        create_scratch_slot, make_test_scheduler, make_test_scheduler_recurrent,
    };

    use super::super::tests_toy_model::tests::ToyState;

    pub(in super::super) fn fork_request(
        scheduler: &mut Scheduler,
        parent: SequenceId,
    ) -> Result<SequenceId, ConversationError> {
        fork_request_at(scheduler, parent, DecodePriority::Normal)
    }

    pub(in super::super) fn fork_request_at(
        scheduler: &mut Scheduler,
        parent: SequenceId,
        band: DecodePriority,
    ) -> Result<SequenceId, ConversationError> {
        let (tx, rx) = flume::bounded(1);
        scheduler.handle_request(SchedulerRequest::ForkEphemeralSequence {
            parent,
            band,
            response_tx: tx,
        });
        rx.recv().expect("scheduler reply")
    }

    /// **A fork decodes at the band its asker named**, whatever its parent's
    /// layer decodes at, and the band goes when the fork is freed.
    #[test]
    fn an_ephemeral_fork_decodes_at_the_band_it_was_asked_at() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let before = scheduler.decode_layer_priority(parent);

        for band in [
            DecodePriority::Low,
            DecodePriority::Normal,
            DecodePriority::High,
        ] {
            let child = fork_request_at(&mut scheduler, parent, band).expect("fork");
            assert_eq!(scheduler.decode_layer_priority(child), Some(band));
            assert_eq!(scheduler.decode_priority_or_high(child), band);
            assert_eq!(
                scheduler.decode_layer_priority(parent),
                before,
                "the parent keeps its own band"
            );
            scheduler.handle_request(SchedulerRequest::FreeSequence { sequence_id: child });
            assert!(!scheduler.slot_bands.contains_key(&child));
        }
    }

    /// **A slot's band can be raised while it lives, and given back.** Set, it
    /// decodes at that band; cleared, at the one its conversation records —
    /// or, recording none, at no band of its own.
    #[test]
    fn a_slots_band_is_set_and_given_back() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = Conversation::new();
        let recorded = ProjectionTarget {
            timeline: TimelineId::from_raw(51).unwrap(),
            layer: LayerId::from_raw(1).unwrap(),
            group: GroupId::from_raw(1).unwrap(),
        };
        conversation
            .set_conversation_metadata(recorded.timeline, DecodePriority::METADATA_KEY, "normal")
            .unwrap();
        conversation.register_timeline(recorded.timeline, recorded.layer, recorded.group);
        let plain = ProjectionTarget {
            timeline: TimelineId::from_raw(52).unwrap(),
            ..recorded
        };
        let a = scheduler
            .create_sequence(conversation.clone(), Some(recorded))
            .expect("slot");
        let b = scheduler
            .create_sequence(conversation, Some(plain))
            .expect("slot");
        for slot in [a, b] {
            scheduler.handle_request(SchedulerRequest::SetBand {
                sequence_id: slot,
                band: Some(DecodePriority::Urgent),
            });
            assert_eq!(
                scheduler.slot_bands.get(&slot),
                Some(&DecodePriority::Urgent)
            );
            scheduler.handle_request(SchedulerRequest::SetBand {
                sequence_id: slot,
                band: None,
            });
        }
        assert_eq!(scheduler.slot_bands.get(&a), Some(&DecodePriority::Normal));
        assert!(!scheduler.slot_bands.contains_key(&b));
    }

    /// **A conversation that records its own band decodes at it**, in place of
    /// its layer's, from the slot its turns run on; the band goes with the
    /// slot. One that records none decodes at its layer's.
    #[test]
    fn a_slot_decodes_at_the_band_its_conversation_records() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = Conversation::new();
        let timeline = TimelineId::from_raw(41).unwrap();
        conversation
            .set_conversation_metadata(timeline, DecodePriority::METADATA_KEY, "normal")
            .unwrap();
        let target = ProjectionTarget {
            timeline,
            layer: LayerId::from_raw(1).unwrap(),
            group: GroupId::from_raw(1).unwrap(),
        };
        conversation.register_timeline(timeline, target.layer, target.group);
        let slot = scheduler
            .create_sequence(conversation.clone(), Some(target))
            .expect("slot");
        assert_eq!(
            scheduler.decode_layer_priority(slot),
            Some(DecodePriority::Normal)
        );
        scheduler.handle_request(SchedulerRequest::FreeSequence { sequence_id: slot });
        assert!(!scheduler.slot_bands.contains_key(&slot));

        let other = ProjectionTarget {
            timeline: TimelineId::from_raw(42).unwrap(),
            ..target
        };
        let plain = scheduler
            .create_sequence(conversation, Some(other))
            .expect("slot");
        assert!(!scheduler.slot_bands.contains_key(&plain));
    }

    /// **A turn run on the fork decodes at the fork's band**: the view's band is
    /// its owning slot's.
    #[test]
    fn a_turn_on_an_ephemeral_fork_decodes_at_the_forks_band() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let child = fork_request_at(&mut scheduler, parent, DecodePriority::Low).expect("fork");
        let view = start_turn_on(&mut scheduler, child);
        assert_eq!(
            scheduler.decode_layer_priority(view),
            Some(DecodePriority::Low)
        );
    }

    /// **An ephemeral fork starts from the live slot's caches and its model
    /// state, and is registered ephemeral.** The projection the question runs
    /// under then keeps the prefix the parent placed, instead of rebuilding it.
    #[test]
    fn an_ephemeral_fork_inherits_the_parents_projection_caches_and_model_state() {
        use projection_assembler::PlacedPiece;

        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let state = scheduler.slot_projection_state.entry(parent).or_default();
        state.placed_pieces.push(PlacedPiece::ending_at(2));
        state.glue_generation = 7;
        let marked: ToyState = [[1.0; 4]; 2];
        probe.set(parent.0, marked);

        let child = fork_request(&mut scheduler, parent).expect("fork");

        assert_ne!(child, parent);
        assert!(
            scheduler.ephemeral_slots.contains(&child),
            "the fork is ephemeral"
        );
        assert!(
            !scheduler.ephemeral_slots.contains(&parent),
            "the parent is not"
        );
        assert!(scheduler.slot_conversations.contains_key(&child));
        assert!(
            !scheduler.turn_views.contains_key(&child),
            "a fork is a slot, not a turn view"
        );
        let forked = &scheduler.slot_projection_state[&child];
        assert_eq!(
            forked.placed_pieces.len(),
            1,
            "the placed record carries over"
        );
        assert_eq!(forked.glue_generation, 7);
        assert_eq!(
            probe.get(child.0),
            Some(marked),
            "recurrent state is forked"
        );
    }

    /// **The parent's in-flight user part stays with the parent.**
    #[test]
    fn an_ephemeral_fork_does_not_take_the_parents_pending_user_part() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        scheduler.slot_projection_state.entry(parent).or_default();

        let child = fork_request(&mut scheduler, parent).expect("fork");
        assert!(scheduler.slot_projection_state[&child]
            .pending_user_part
            .is_none());
    }

    /// **Freeing the fork leaves the parent as it was**, and the fork's own
    /// bookkeeping goes with it.
    #[test]
    fn freeing_an_ephemeral_fork_leaves_the_parent_intact() {
        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let marked: ToyState = [[2.0; 4]; 2];
        probe.set(parent.0, marked);
        let child = fork_request(&mut scheduler, parent).expect("fork");

        scheduler.handle_request(SchedulerRequest::FreeSequence { sequence_id: child });

        assert!(!scheduler.ephemeral_slots.contains(&child));
        assert!(!scheduler.slot_conversations.contains_key(&child));
        assert!(!scheduler.slot_projection_state.contains_key(&child));
        assert_eq!(
            probe.get(parent.0),
            Some(marked),
            "the parent's state survives"
        );
        assert!(scheduler.slot_conversations.contains_key(&parent));
    }

    /// A parent with a turn in flight: a view carved from it is registered.
    pub(in super::super) fn start_turn_on(
        scheduler: &mut Scheduler,
        parent: SequenceId,
    ) -> SequenceId {
        let conversation = crate::projection::Conversation::new();
        let view = create_scratch_slot(scheduler, &conversation);
        scheduler.turn_views.insert(
            view,
            ViewState {
                parent_id: parent,
                original_borrowed: BlockCount(0),
                turn_start_parent_blocks: 0,
                question_tokens: 0,
                applied_identity: None,
                state_forked: true,
                holds_parent_state: false,
            },
        );
        view
    }

    pub(in super::super) fn park_fork(
        scheduler: &mut Scheduler,
        parent: SequenceId,
    ) -> flume::Receiver<Result<SequenceId, ConversationError>> {
        let (tx, rx) = flume::bounded(1);
        scheduler.handle_request(SchedulerRequest::ForkEphemeralSequence {
            parent,
            band: DecodePriority::Low,
            response_tx: tx,
        });
        rx
    }

    /// **The fork itself is refused while a turn runs on the parent**: its view
    /// is borrowing the blocks the fork would carve.
    #[test]
    fn an_ephemeral_fork_is_never_taken_while_a_turn_runs_on_the_parent() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        start_turn_on(&mut scheduler, parent);

        let err = scheduler
            .fork_ephemeral_sequence(parent, DecodePriority::Normal)
            .expect_err("refused");
        assert!(
            matches!(err, ConversationError::TurnInFlight { .. }),
            "{err:?}"
        );
    }

    /// **A request that arrives mid-turn waits for the turn boundary**, then is
    /// answered with a fork of the finished state, with nothing for the caller to
    /// lock.
    #[test]
    fn a_fork_requested_mid_turn_is_answered_when_the_turn_ends() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let view = start_turn_on(&mut scheduler, parent);

        let rx = park_fork(&mut scheduler, parent);
        assert!(rx.try_recv().is_err(), "no answer while the turn runs");
        scheduler.drain_parked_forks();
        assert!(rx.try_recv().is_err(), "still parked while the turn runs");

        scheduler.turn_views.remove(&view);
        scheduler.drain_parked_forks();
        let child = rx
            .try_recv()
            .expect("answered at the boundary")
            .expect("fork");
        assert!(scheduler.ephemeral_slots.contains(&child));
        assert!(scheduler.parked_forks.is_empty());
        assert_eq!(
            scheduler.decode_layer_priority(child),
            Some(DecodePriority::Low),
            "a parked request keeps the band it was asked at"
        );
    }

    /// **Two requests parked on one parent get a fork each.**
    #[test]
    fn requests_parked_on_one_parent_each_get_their_own_fork() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let view = start_turn_on(&mut scheduler, parent);
        let (a, b) = (
            park_fork(&mut scheduler, parent),
            park_fork(&mut scheduler, parent),
        );

        scheduler.turn_views.remove(&view);
        scheduler.drain_parked_forks();
        let (a, b) = (
            a.try_recv().unwrap().unwrap(),
            b.try_recv().unwrap().unwrap(),
        );
        assert_ne!(a, b);
    }

    /// **A parent freed while a request waits fails it**, so the caller does not
    /// hang.
    #[test]
    fn a_parked_fork_fails_when_its_parent_is_freed() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        start_turn_on(&mut scheduler, parent);
        let rx = park_fork(&mut scheduler, parent);

        scheduler.handle_request(SchedulerRequest::FreeSequence {
            sequence_id: parent,
        });
        scheduler.drain_parked_forks();
        assert!(rx.try_recv().expect("answered").is_err());
    }

    /// **The fork shares no write target with its parent.** It borrows the
    /// parent's blocks read-only and writes into chunks of its own: cutting the
    /// fork back to nothing, then freeing it, leaves the parent's blocks and
    /// offset exactly as they were.
    #[test]
    fn an_ephemeral_fork_shares_no_write_target_with_its_parent() {
        let (mut scheduler, _tx, _probe) = make_test_scheduler_recurrent();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let tokens = [1u32, 2, 3, 4];
        let input = Tensor::new(&tokens[..], &scheduler.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        scheduler
            .session
            .ensure_capacity(&[parent.0], tokens.len())
            .unwrap();
        let nl = scheduler.model.num_layers().max(1);
        scheduler
            .model
            .forward_wave(
                &mut scheduler.session,
                &[],
                &[],
                &[parent.0],
                &[input],
                &[],
                &[],
                0,
                nl,
                None,
            )
            .unwrap();
        scheduler
            .session
            .advance_sequence(parent.0, tokens.len())
            .unwrap();
        let blocks = scheduler.session.sequence_block_count(parent.0).unwrap();
        let offset = scheduler.session.sequence_offset(parent.0).unwrap();
        assert!(blocks > 0, "the parent owns KV");

        let child = fork_request(&mut scheduler, parent).expect("fork");
        assert_eq!(
            scheduler.session.sequence_block_count(child.0),
            Some(blocks + 1),
            "the fork sees the parent's prefix, plus a writer chunk of its own"
        );

        scheduler
            .session
            .truncate_sequence_to_blocks(child.0, 0)
            .unwrap();
        assert_eq!(
            scheduler.session.sequence_block_count(parent.0),
            Some(blocks)
        );

        scheduler.handle_request(SchedulerRequest::FreeSequence { sequence_id: child });
        assert_eq!(
            scheduler.session.sequence_block_count(parent.0),
            Some(blocks)
        );
        assert_eq!(scheduler.session.sequence_offset(parent.0), Some(offset));
    }

    /// **A parent with no registered conversation cannot be forked.**
    #[test]
    fn an_ephemeral_fork_of_an_unknown_slot_is_refused() {
        let (mut scheduler, _tx) = make_test_scheduler();
        assert!(fork_request(&mut scheduler, SequenceId(99)).is_err());
    }
}
