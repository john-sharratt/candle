//! Tests for whole turns through the loop: submit, prefill, decode and seal,
//! including abandoned, failed and replayed turns.
//!
//! A turn that runs out of budget streams every token it generated. A turn with
//! no decode budget replies nothing, since the token its prefill sampled was
//! never forwarded. Consecutive turns carry the parent's recurrent buffer forward,
//! which is checked by an exact affine identity. A submit over an abandoned turn
//! winds that turn down first, while a submit over a turn still prefilling is
//! refused. A failed prefill releases its view, so the parent can be submitted to
//! again.

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

    use candle::{DType, Tensor};

    use std::time::Duration;

    use super::super::tests_fixtures::tests::{
        create_scratch_slot, forward_on, make_test_scheduler, make_test_scheduler_recurrent,
    };

    use super::super::tests_toy_model::tests::ToyState;

    use super::super::tests_admission::tests::admission_scheduler;

    use super::super::tests_pages::tests::boundary_state;

    /// A minimal raw-path SubmitTurn (no projection, no substrate write) —
    /// the request shape the abandoned-turn wind-down tests drive.
    pub(in super::super) fn raw_submit_turn(
        parent: SequenceId,
        event_tx: Sender<TurnEvent>,
    ) -> SchedulerRequest {
        SchedulerRequest::SubmitTurn {
            seal_group: None,
            sequence_id: parent,
            projection_inputs: None,
            prefill_tokens: TokenBuffer::from(vec![1u32, 2, 3]),
            prefill_text: String::new(),
            user_text: String::new(),
            tags: Vec::new(),
            user_content_start: 0,
            user_content_end: 0,
            assistant_content_start: 0,
            no_think: false,
            projection_offsets: Vec::new(),
            prefill_assistant_text: String::new(),
            post_decode_tokens: TokenBuffer::default(),
            max_decode_tokens: 4,
            sampling: SamplingConfig::default(),
            event_tx,
            reprojection: None,
            disable_reprojection: false,
            triggers: Arc::new(TriggerRegistry::new()),
            turn_grammar: None,
            free_tool_calls_from_penalties: false,
            recorded_reply: None,
            recurrent_spans: Vec::new(),
        }
    }

    /// The wind-down half of the abandoned-turn contract: a `SubmitTurn` that
    /// finds this parent's previous turn still registered — its handle
    /// dropped, its event channel closed — finishes that turn through the
    /// ordinary cleanup BEFORE the new turn touches the parent, so two
    /// decodes never share a slot.
    #[test]
    fn a_submit_over_an_abandoned_turn_winds_it_down_first() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let view = create_scratch_slot(&mut scheduler, &conversation);

        // The abandoned turn: a decode whose caller dropped its handle.
        let (state, view_rx) = boundary_state();
        drop(view_rx);
        scheduler.active_decodes.insert(view, state);
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

        let (event_tx, event_rx) = flume::unbounded();
        assert!(scheduler.handle_request(raw_submit_turn(parent, event_tx)));

        // The abandoned decode was finalized (and the new turn has not begun
        // decoding — it is queued as prefill), so no decode remains. The
        // freed view slot's id may be recycled by the NEW turn's view, so the
        // assertions observe the contract, not the slot id.
        assert!(
            scheduler.active_decodes.is_empty(),
            "the abandoned decode must be finalized before the new turn runs"
        );
        assert_eq!(
            scheduler.pending_submits.len(),
            1,
            "the new turn must wait for admission, its projection priced"
        );
        assert!(
            !matches!(
                event_rx.try_recv(),
                Ok(TurnEvent::Error(ConversationError::TurnInFlight { .. }))
            ),
            "the new turn must not be refused"
        );
    }

    /// **A turn that runs out of budget streams every token it generated, the
    /// last included.** Only an end-of-sequence token goes unstreamed; the final
    /// token of a length-capped turn is an ordinary token, and a caller counting
    /// what it received must see the same number the turn reports.
    ///
    /// Driven through the whole loop — submit, prefill, decode, cleanup — since
    /// the streaming decision sits on both the prefill's first token and the
    /// decode commit, and only the loop runs them in the order a caller sees.
    #[test]
    fn a_length_capped_turn_streams_every_token_it_generated() {
        let (mut scheduler, tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let loop_thread = std::thread::spawn(move || scheduler.run());

        let (event_tx, event_rx) = flume::unbounded();
        let mut submit = raw_submit_turn(parent, event_tx);
        if let SchedulerRequest::SubmitTurn { sampling, .. } = &mut submit {
            // The dummy model's logits are flat, so the end-of-sequence token
            // would otherwise be drawn now and then; banned, the turn can only
            // end on its budget.
            sampling.banned_tokens = vec![0];
        }
        tx.send(submit).expect("scheduler running");
        let mut streamed = Vec::new();
        let done = loop {
            match event_rx
                .recv_timeout(Duration::from_secs(60))
                .expect("the turn ends within a minute")
            {
                TurnEvent::Token(t) => streamed.push(t),
                TurnEvent::Done(resp) => break resp,
                TurnEvent::Error(e) => panic!("the turn failed: {e}"),
                _ => {}
            }
        };
        tx.send(SchedulerRequest::Shutdown)
            .expect("scheduler running");
        loop_thread.join().expect("the loop exits cleanly");

        assert_eq!(done.stats.finish, FinishReason::Length);
        assert_eq!(done.token_ids.len(), 4, "max_decode_tokens is 4");
        assert_eq!(streamed, done.token_ids.to_vec());
    }

    /// **A turn that decodes nothing replies nothing.** The prefill samples a
    /// token after its last position whatever the budget, but with a budget of
    /// zero that token is never forwarded: it is the model's guess at what
    /// would come next. It must not reach the stream, the response, or — since
    /// the seal builds the turn's stored text from the response — the substrate,
    /// where a prefilled line would read as itself plus the first word of a
    /// sentence nobody wrote.
    #[test]
    fn a_turn_with_no_decode_budget_reports_no_token() {
        let (mut scheduler, tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let loop_thread = std::thread::spawn(move || scheduler.run());

        let (event_tx, event_rx) = flume::unbounded();
        let mut submit = raw_submit_turn(parent, event_tx);
        if let SchedulerRequest::SubmitTurn {
            max_decode_tokens,
            sampling,
            ..
        } = &mut submit
        {
            *max_decode_tokens = 0;
            // Not end-of-sequence, so the sampled token is an ordinary word.
            sampling.banned_tokens = vec![0];
        }
        tx.send(submit).expect("scheduler running");
        let mut streamed = Vec::new();
        let done = loop {
            match event_rx
                .recv_timeout(Duration::from_secs(60))
                .expect("the turn ends within a minute")
            {
                TurnEvent::Token(t) => streamed.push(t),
                TurnEvent::Done(resp) => break resp,
                TurnEvent::Error(e) => panic!("the turn failed: {e}"),
                _ => {}
            }
        };
        tx.send(SchedulerRequest::Shutdown)
            .expect("scheduler running");
        loop_thread.join().expect("the loop exits cleanly");

        assert!(
            streamed.is_empty(),
            "streamed {streamed:?} for a turn that decodes nothing"
        );
        assert!(
            done.token_ids.is_empty(),
            "reported {:?} for a turn that decodes nothing",
            done.token_ids
        );
        assert_eq!(done.text, "");
        assert_eq!(done.stats.tokens_generated, 0);
    }

    /// **Consecutive turns on one parent carry its recurrent buffer forward.**
    ///
    /// A recurrence cannot be rebuilt from K/V, so between one turn and the next
    /// the parent's state is the only copy of everything the conversation has
    /// said — the seal empties the slot's K/V (stateless-slot housekeeping) and
    /// keeps the state. Three identical turns are driven through the whole loop
    /// — submit, prefill, view carve, decode, move back, seal — and the parent
    /// is read after each.
    ///
    /// The toy step is affine (`s ← 2s + t·k`), so a turn of `m` forwards is
    /// `f(s) = 2ᵐ·s + C`, and `C = f(0)` is exactly the state after turn one.
    /// Carried forward therefore means `state₂ = 2ᵐ·state₁ + state₁` and
    /// `state₃ = 2ᵐ·state₂ + state₁` for one `m` — an exact identity, no
    /// tolerance. A state that was reset or not reused comes back as `state₁`;
    /// one absorbed twice, or out of order, matches no `m` at all.
    #[test]
    fn consecutive_turns_carry_the_recurrent_buffer_forward() {
        let (mut scheduler, tx, probe) = make_test_scheduler_recurrent();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let loop_thread = std::thread::spawn(move || scheduler.run());

        // One turn, run to its end. `Done` is sent from the seal, which runs
        // after the view's state has moved back onto the parent and the closing
        // tail has been forwarded, so the parent is settled when this returns.
        let run_turn = || {
            let (event_tx, event_rx) = flume::unbounded();
            let mut submit = raw_submit_turn(parent, event_tx);
            if let SchedulerRequest::SubmitTurn { sampling, .. } = &mut submit {
                // Flat dummy logits draw end-of-sequence now and then; banned,
                // every turn runs its whole budget and the turns are identical.
                sampling.banned_tokens = vec![0];
            }
            tx.send(submit).expect("scheduler running");
            loop {
                match event_rx
                    .recv_timeout(Duration::from_secs(60))
                    .expect("the turn ends within a minute")
                {
                    TurnEvent::Done(resp) => break resp,
                    TurnEvent::Error(e) => panic!("the turn failed: {e}"),
                    _ => {}
                }
            }
        };
        let read = || {
            (
                probe
                    .get_index(parent.0)
                    .expect("the parent carries a recurrence after a turn")
                    .tokens,
                probe
                    .get(parent.0)
                    .expect("the parent carries a recurrent state after a turn"),
            )
        };

        let first = run_turn();
        let (absorbed_1, state_1) = read();
        let second = run_turn();
        let (absorbed_2, state_2) = read();
        let third = run_turn();
        let (absorbed_3, state_3) = read();

        tx.send(SchedulerRequest::Shutdown)
            .expect("scheduler running");
        loop_thread.join().expect("the loop exits cleanly");

        // Each turn's K/V is that turn alone (the seal empties the slot), and the
        // recurrence absorbs exactly that many tokens per turn.
        for (turn, held) in [
            (1, first.stats.context_tokens),
            (2, second.stats.context_tokens),
            (3, third.stats.context_tokens),
        ] {
            assert_eq!(
                held, absorbed_1 as usize,
                "turn {turn}'s K/V holds {held} token(s), turn 1 absorbed {absorbed_1}",
            );
        }
        assert_eq!(
            (absorbed_2 - absorbed_1, absorbed_3 - absorbed_2),
            (absorbed_1, absorbed_1),
            "every identical turn must absorb the same {absorbed_1} token(s); the counts \
             went {absorbed_1} → {absorbed_2} → {absorbed_3}",
        );

        // `f(s) = 2ᵐ·s + state₁`: recover `2ᵐ` from one cell and hold every cell
        // of both later turns to it.
        let scale = (state_2[0][0] - state_1[0][0]) / state_1[0][0];
        assert!(
            scale >= 2.0 && scale.log2().fract() == 0.0,
            "turn 2 did not continue from turn 1's state: state₂[0][0] = {} against \
             state₁[0][0] = {} is no 2ᵐ step (scale {scale}). state₁ = {state_1:?}, \
             state₂ = {state_2:?}",
            state_2[0][0],
            state_1[0][0],
        );
        let step = |from: &ToyState| -> ToyState {
            let mut out = *from;
            for (l, row) in out.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell = *cell * scale + state_1[l][j];
                }
            }
            out
        };
        assert_eq!(
            state_2,
            step(&state_1),
            "turn 2 did not continue from turn 1's state — the parent's recurrent \
             buffer was not carried into the second turn",
        );
        assert_eq!(
            state_3,
            step(&state_2),
            "turn 3 did not continue from turn 2's state — the parent's recurrent \
             buffer was not carried into the third turn",
        );
    }

    /// **A turn group's cases are masked from each other.** Every case starts
    /// from the prefix's recurrent state, not from the end of the case before
    /// it, so the state a group leaves on the parent — its last case's — is
    /// exactly what a lone prefill of that case leaves.
    ///
    /// The toy recurrence is affine and absorbs every token (`s ← 2s + t·k`),
    /// so a case that had run on from its predecessors would carry their tokens
    /// in its state and could not match the lone run. Driven through the whole
    /// loop — queue, carve at admission, batched prefill, in-order seal,
    /// release — so a case carved late must still fork the prefix's state.
    #[test]
    fn a_turn_groups_cases_each_start_from_the_prefix() {
        let run_group = |cases: &[&[u32]]| {
            let (mut scheduler, tx, probe) = make_test_scheduler_recurrent();
            let conversation = crate::projection::Conversation::new();
            let parent = create_scratch_slot(&mut scheduler, &conversation);
            forward_on(&mut scheduler, parent.0, &[5, 6, 7, 8]);
            let prefix = probe.get(parent.0).expect("the prefix left a state");
            let blocks = scheduler.session.sequence_block_count(parent.0).unwrap();
            let (event_tx, event_rx) = flume::unbounded();
            scheduler.start_turn_group(
                parent,
                vec![BlockRange::new(0, blocks)],
                cases
                    .iter()
                    .map(|tokens| TurnContent {
                        token_ids: TokenBuffer::from(tokens.to_vec()),
                        ..TurnContent::default()
                    })
                    .collect(),
                SamplingConfig::default(),
                PriorBelief::default(),
                event_tx,
            );
            let loop_thread = std::thread::spawn(move || scheduler.run());
            // **This CPU session cannot snapshot its K/V** (the stub model writes
            // none, so a seal finds an empty range), so no case seals and the
            // group answers that it lost them all — what a group that lost every
            // case must say. Every case still ran its prefill and released its
            // state in order, which is what this test reads; a group answering
            // `Done` is [`turn_group_verdict`]'s test.
            let answer = loop {
                match event_rx
                    .recv_timeout(Duration::from_secs(60))
                    .expect("the group ends within a minute")
                {
                    TurnEvent::Done(_) => panic!("a group whose cases sealed nothing is not done"),
                    TurnEvent::Error(e) => break e.to_string(),
                    _ => {}
                }
            };
            assert!(
                answer.ends_with(&format!("0 of {n} cases sealed, {n} lost", n = cases.len())),
                "the group names every case it could not seal: {answer}",
            );
            tx.send(SchedulerRequest::Shutdown)
                .expect("scheduler running");
            loop_thread.join().expect("the loop exits cleanly");
            let after = probe
                .get(parent.0)
                .expect("the last case's state moved onto the parent");
            (prefix, after)
        };

        let last: &[u32] = &[21, 22, 23];
        let (prefix, alone) = run_group(&[last]);
        assert_ne!(alone, prefix, "the lone case advanced the prefix's state");
        let (_, grouped) = run_group(&[&[11, 12, 13, 14, 15], &[31, 32], last]);
        assert_eq!(
            grouped, alone,
            "the group's last case must leave exactly what a lone prefill of it leaves — \
             a different state means it ran on from the cases before it",
        );
    }

    /// The refusal half: a previous turn that is registered but has no decode
    /// to finish (still in prefill) cannot be wound down — the new submit is
    /// refused with `TurnInFlight`, and the in-flight turn is left alone.
    #[test]
    fn a_submit_over_a_still_prefilling_turn_is_refused() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let view = create_scratch_slot(&mut scheduler, &conversation);

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

        let (event_tx, event_rx) = flume::unbounded();
        assert!(scheduler.handle_request(raw_submit_turn(parent, event_tx)));

        assert!(
            matches!(
                event_rx.try_recv(),
                Ok(TurnEvent::Error(ConversationError::TurnInFlight { .. }))
            ),
            "a submit racing a still-prefilling turn must be refused"
        );
        assert!(
            scheduler.prefill_queue.is_empty(),
            "the refused turn must queue nothing"
        );
        assert!(
            scheduler.turn_views.contains_key(&view),
            "the refused submit must not disturb the in-flight turn"
        );
    }

    /// The errored-prefill counterpart to the dropped-handle wind-down: a turn
    /// view whose prefill FAILED (its error surfaced and it was drained from
    /// `active_prefills`) must be let go, or the parent's next submit is refused
    /// with `TurnInFlight` forever — a character alive and scheduled that never
    /// acts again after one faulted wave. This is the regression for that wedge.
    #[test]
    fn a_failed_prefill_view_is_reclaimed_so_the_parent_is_submittable() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let conversation = crate::projection::Conversation::new();
        let parent = create_scratch_slot(&mut scheduler, &conversation);
        let view = create_scratch_slot(&mut scheduler, &conversation);

        // The view was registered before prefill; the prefill then faulted and
        // was drained, which discards the view.
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
        scheduler.discard_turn_view(view);
        assert!(
            !scheduler.turn_views.contains_key(&view),
            "the failed view must be let go, not left dangling under its parent"
        );

        // The parent is submittable again — not the permanent TurnInFlight the
        // dangling view produced.
        let (event_tx, event_rx) = flume::unbounded();
        assert!(scheduler.handle_request(raw_submit_turn(parent, event_tx)));
        assert!(
            !matches!(
                event_rx.try_recv(),
                Ok(TurnEvent::Error(ConversationError::TurnInFlight { .. }))
            ),
            "a submit after a failed view is reclaimed must not be refused"
        );
        assert_eq!(
            scheduler.pending_submits.len(),
            1,
            "the new turn must wait for admission, its projection priced"
        );
    }

    /// **A replayed turn commits its recording, whatever the sampler chose**,
    /// then the end of turn once the recording is spent.
    #[test]
    fn a_replayed_turn_commits_its_recording_in_place_of_the_sample() {
        let (mut scheduler, _tx, _probe) = make_test_scheduler_recurrent();
        let slot = SequenceId(scheduler.session.create_sequence().unwrap());
        let (mut state, _rx) = boundary_state();
        state.recorded_reply = Some(Replay::new(vec![7, 8]));
        scheduler.active_decodes.insert(slot, state);
        let row = Tensor::zeros(16, DType::F32, &scheduler.device).unwrap();
        for _ in 0..3 {
            scheduler.commit_decoded_tokens(&[slot], &mut [3], std::slice::from_ref(&row));
        }
        let state = &scheduler.active_decodes[&slot];
        assert_eq!(&state.generated_tokens[..], &[7u32, 8, 0][..]);
        assert!(
            state.finished,
            "the end of turn after the recording finishes it"
        );
    }

    /// **A turn-group case queues without a view, and is carved one when
    /// admission takes it** — not before, so nothing is claimed for a case the
    /// engine has not judged it can carry.
    #[test]
    fn a_group_case_is_carved_its_view_only_when_admitted() {
        let mut sched = admission_scheduler();
        let parent = SequenceId(sched.session.create_sequence().unwrap());
        let (event_tx, _event_rx) = flume::unbounded();
        let case = || TurnContent {
            token_ids: TokenBuffer::from(vec![1u32; 64]),
            ..TurnContent::default()
        };
        sched.start_turn_group(
            parent,
            Vec::new(),
            vec![case(), case()],
            SamplingConfig::default(),
            PriorBelief::default(),
            event_tx,
        );
        assert_eq!(sched.prefill_queue.len(), 2, "both cases queued");
        assert!(
            sched.prefill_queue.iter().all(|w| w.sequence_id == parent),
            "on the parent's slot: no view carved yet",
        );
        assert!(
            sched.turn_views.is_empty(),
            "nothing claimed before admission"
        );

        sched.promote_new_prefills();

        let started = &sched.active_prefills[0].work;
        assert_ne!(
            started.sequence_id, parent,
            "admission carved the case its own view"
        );
        assert!(sched.turn_views.contains_key(&started.sequence_id));
    }

    /// **Resetting a parent takes its group's queued cases with it.** They wait
    /// on the parent's own slot for a view; left behind with the group gone,
    /// one would be prefilled straight into whatever the reset slot holds next.
    #[test]
    fn resetting_a_parent_drops_its_groups_queued_cases() {
        let mut sched = admission_scheduler();
        let parent = SequenceId(sched.session.create_sequence().unwrap());
        let (event_tx, event_rx) = flume::unbounded();
        let case = || TurnContent {
            token_ids: TokenBuffer::from(vec![1u32; 64]),
            ..TurnContent::default()
        };
        sched.start_turn_group(
            parent,
            Vec::new(),
            vec![case(), case(), case()],
            SamplingConfig::default(),
            PriorBelief::default(),
            event_tx,
        );
        assert_eq!(sched.prefill_queue.len(), 3);

        let (tx, rx) = flume::bounded(1);
        sched.handle_request(SchedulerRequest::ResetSequence {
            sequence_id: parent,
            response_tx: tx,
        });
        rx.recv().unwrap().expect("reset");

        assert!(
            sched.prefill_queue.is_empty(),
            "a queued case outlived its group"
        );
        assert!(matches!(event_rx.try_recv(), Ok(TurnEvent::Error(_))));
    }
}
