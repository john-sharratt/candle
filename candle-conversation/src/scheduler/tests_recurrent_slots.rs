//! Tests for the recurrent state's slot lifecycle: a recycled slot never inherits
//! another conversation's memory, and a discarded view hands its state back.
//!
//! Freeing a slot releases its state, including the seeded mark, so a recycled id
//! neither inherits the previous conversation's memory nor skips a genuine reset.
//! The disposition rule is checked as a pair: a mid-turn reprojection moves the
//! state with the K/V, and a clean reprefill discards it. A parked slot whose
//! record cannot be read is an error, and it never starts again from the
//! sequence-start value. One turn is priced at one store however many are parked,
//! so an empty engine never divides by zero.

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

    use candle_transformers::models::speculative_choice::GreedyChooser;

    use candle_transformers::models::batched_inference::{
        BatchedInferenceSession, ManagedBatchedModel,
    };

    use candle_transformers::models::wave_driver::{StepOutcome, StepSpec};

    use super::super::tests_fixtures::tests::{
        forward_on, make_test_scheduler_recurrent, make_test_session, stub_residual, stub_step,
        DummyModel,
    };

    use super::super::tests_toy_model::tests::{
        toy_advance, DummyRecurrentModel, RecurrentProbe, ToyState, ZERO_STATE,
    };

    /// Exact state equality — no tolerance, ever.
    ///
    /// A tolerance here would pass a state that is plumbed correctly but
    /// *wrong*, which is the entire failure class this document's tier 1
    /// exists to catch.
    #[track_caller]
    pub(in super::super) fn assert_state_eq(
        probe: &RecurrentProbe,
        seq: usize,
        expected: ToyState,
    ) {
        let actual = probe
            .get(seq)
            .unwrap_or_else(|| panic!("sequence {seq} carries no recurrent state at all"));
        assert_eq!(actual, expected, "recurrent state for sequence {seq}");
    }

    // —— Tests ————————————————————————————————————————————————————————————————

    // —— recurrent release-on-free tests ——————————————————————————————————————

    /// **The recycled-slot defect.** Slot ids are pool indices and come back
    /// round; the recurrent map is keyed by that id and owned by the model,
    /// which outlives every session. So a freed slot whose state was never
    /// released leaves a stranger's memory sitting under an id the next
    /// conversation is about to be handed.
    ///
    /// The second sequence's first wave carries `offset > 0` — it is resuming
    /// against injected K/V, which is the normal case and precisely the one
    /// `ensure_recurrent`'s `offset == 0` reset cannot catch. Without the
    /// release hook the `Occupied` arm hands back the previous conversation's
    /// state, every shape matches, nothing errors, and the model answers from
    /// a history this cache never had.
    #[test]
    fn a_recycled_slot_does_not_inherit_the_previous_conversations_state() {
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };

        // Conversation A runs on slot 3 and leaves a non-zero state behind.
        probe.ensure(3, 0);
        probe
            .states
            .lock()
            .unwrap()
            .insert(3, [[1.0, 2.0, 3.0, 4.0]; 2]);
        assert_ne!(probe.get(3), Some(ZERO_STATE), "A left something behind");

        // The slot is freed. This is the call P0.3 wires into every free path.
        model.release_sequence(3).unwrap();
        assert_eq!(
            probe.get(3),
            None,
            "the freed slot must carry no state at all"
        );

        // Conversation B lands on the recycled id, and its first wave is a
        // resume: offset > 0, so the reset does not fire.
        probe.ensure(3, 64);
        assert_eq!(
            probe.get(3),
            Some(ZERO_STATE),
            "a recycled slot inherited the previous conversation's recurrent \
             state — the model now remembers a conversation this cache never had"
        );
    }

    /// The leak half of the same defect: the map is the model's, so nothing
    /// else can shrink it. At ~63 MiB per conversation on the 35B, a map that
    /// only grows is the process's lifetime memory profile.
    #[test]
    fn releasing_a_sequence_returns_the_map_to_its_previous_size() {
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };
        probe.ensure(0, 0);
        let before = probe.len();

        probe.ensure(1, 0);
        probe.ensure(2, 0);
        assert_eq!(probe.len(), before + 2);

        model.release_sequence(1).unwrap();
        model.release_sequence(2).unwrap();
        assert_eq!(
            probe.len(),
            before,
            "freed sequences must not stay resident in the recurrent map"
        );
    }

    /// Releasing clears the seeded mark too. A stale mark would make the next
    /// conversation on that id skip its `offset == 0` reset exactly once —
    /// which is the recycled-slot defect wearing the fix's own clothes.
    #[test]
    fn releasing_a_sequence_clears_its_seeded_mark() {
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };
        probe.ensure(5, 0);
        probe.seeded.lock().unwrap().insert(5);
        model.release_sequence(5).unwrap();
        assert!(!probe.is_seeded(5), "the seeded mark outlived the sequence");

        probe.ensure(5, 0);
        probe.set(5, [[9.0; 4]; 2]);
        probe.ensure(5, 0);
        assert_eq!(
            probe.get(5),
            Some(ZERO_STATE),
            "a stale seeded mark suppressed the reset for a new conversation"
        );
    }

    // —— view state-carry tests (the switch gate) —————————————————————————————

    /// **The gate.** A view is a distinct sequence id, so without an explicit
    /// carry the recurrent state does not cross a turn boundary in *either*
    /// direction: the child starts from the sequence-start value while its K/V
    /// borrows the parent's whole history, and at finalize only the blocks go
    /// back.
    ///
    /// Three turns must land somewhere a one-turn conversation does not. The
    /// toy recurrence is order-dependent, so this fails both ways — a lost
    /// turn and a duplicated one land on different values.
    #[test]
    fn state_advances_across_turn_boundaries() {
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };

        // One "turn": carve a view off the parent, advance it, finalize.
        let turn = |parent: usize, view: usize, tokens: usize, offset: usize| {
            model.fork_recurrent(parent, view).unwrap();
            probe.ensure(view, offset);
            let mut s = probe.get(view).expect("the view carries state");
            toy_advance(&mut s, tokens);
            probe.set(view, s);
            model.move_recurrent(view, parent).unwrap();
        };

        probe.ensure(0, 0);
        turn(0, 1, 5, 0);
        let after_one = probe.get(0).unwrap();

        turn(0, 1, 5, 64);
        turn(0, 1, 5, 128);
        let after_three = probe.get(0).unwrap();

        assert_ne!(
            after_three, after_one,
            "the recurrent state did not advance across turn boundaries — on a \
             hybrid this is three quarters of the stack contributing nothing, \
             and it reads perfectly"
        );
        // s ← 2s + 5k, three times from zero: 5k·(4+2+1) = 35k.
        assert_eq!(after_three[0][0], 35.0, "turn 1 then 2 then 3, in order");
        assert_eq!(after_one[0][0], 5.0);
    }

    /// **The disposition pair — both halves required.**
    ///
    /// A build that always moves passes (a) and fails (b); one that always
    /// discards passes (b) and fails (a). Only the pair pins the rule, and the
    /// rule is: *the recurrent state follows the K/V*.
    #[test]
    fn view_disposition_moves_on_reprojection_and_discards_on_clean_reprefill() {
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };

        // (a) MID-TURN REPROJECTION MOVES. The KV is preserved — the tail is
        // captured and re-injected — so the state must survive with it. Decode
        // 10, reproject, decode 10; that has to equal a straight-through 20.
        probe.ensure(0, 0);
        model.fork_recurrent(0, 1).unwrap();
        probe.ensure(1, 0);
        {
            let mut s = probe.get(1).unwrap();
            toy_advance(&mut s, 10);
            probe.set(1, s);
        }
        // Reprojection: the view finalizes onto the parent, then a fresh view
        // is carved off it.
        model.move_recurrent(1, 0).unwrap();
        model.fork_recurrent(0, 2).unwrap();
        probe.ensure(2, 64);
        {
            let mut s = probe.get(2).unwrap();
            toy_advance(&mut s, 10);
            probe.set(2, s);
        }
        model.move_recurrent(2, 0).unwrap();
        let reprojected = probe.get(0).unwrap();

        let mut straight = ZERO_STATE;
        toy_advance(&mut straight, 10);
        toy_advance(&mut straight, 10);
        assert_eq!(
            reprojected, straight,
            "a reprojection discarded the state — the turn's first ten tokens \
             are gone and the reply will read as though it never saw them"
        );

        // (b) CLEAN REPREFILL DISCARDS. The thinking tokens are being un-said:
        // the view's blocks are abandoned and re-prefilled clean onto the
        // untouched parent, so the state must be abandoned with them.
        let probe2 = RecurrentProbe::default();
        let model2 = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe2.clone(),
        };
        probe2.ensure(0, 0);
        model2.fork_recurrent(0, 1).unwrap();
        probe2.ensure(1, 0);
        {
            // The turn decodes WITH thinking.
            let mut s = probe2.get(1).unwrap();
            toy_advance(&mut s, 30);
            probe2.set(1, s);
        }
        // Clean seal: do NOT move. Discard the view, re-prefill clean onto the
        // parent, which never saw the thinking tokens.
        model2.release_sequence(1).unwrap();
        {
            let mut s = probe2.get(0).unwrap();
            toy_advance(&mut s, 8); // the clean re-prefill
            probe2.set(0, s);
        }
        let after_thinking_turn = probe2.get(0).unwrap();

        let mut never_thought = ZERO_STATE;
        toy_advance(&mut never_thought, 8);
        assert_eq!(
            after_thinking_turn, never_thought,
            "a thinking turn's sealed state absorbed the reasoning tokens as \
             well as the clean ones — the state saw [think][clean] while the \
             K/V holds only [clean], and it compounds every turn"
        );
    }

    /// A move leaves nothing under the child id. A lingering entry is a leak
    /// keyed by a slot index that is about to be recycled, which is defect 6
    /// re-entering through the fix for defect 0.
    #[test]
    fn move_recurrent_leaves_no_entry_under_the_child() {
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };
        probe.ensure(0, 0);
        probe.set(0, [[3.0; 4]; 2]);
        model.fork_recurrent(0, 1).unwrap();
        assert!(probe.get(1).is_some());

        model.move_recurrent(1, 0).unwrap();
        assert_eq!(probe.get(1), None, "the child's entry outlived the move");
        assert!(!probe.is_seeded(1), "and so did its seeded mark");
        assert_eq!(
            probe.get(0),
            Some([[3.0; 4]; 2]),
            "the parent has the state"
        );
    }

    /// **A view a reprojection re-carved holds the parent's state, and a
    /// discard hands it back.** The rebuild moves the turn's store onto the
    /// view instead of copying it, so the parent has none while the turn runs;
    /// releasing the view with its state would leave the parent's next turn
    /// starting from a store with none of the conversation in it.
    #[test]
    fn a_discarded_recarved_view_hands_the_parents_state_back() {
        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
        let parent = scheduler.session.create_sequence().unwrap();
        forward_on(&mut scheduler, parent, &[1, 2, 3, 4]);
        let held = probe.get(parent).expect("the parent carries state");

        let (view, borrowed) = scheduler
            .carve_view(SequenceId(parent), &[])
            .expect("view carve");
        scheduler.model.move_recurrent(parent, view.0).unwrap();
        assert_eq!(probe.get(parent), None, "moved, not copied");
        scheduler.turn_views.insert(
            view,
            ViewState {
                parent_id: SequenceId(parent),
                original_borrowed: borrowed,
                turn_start_parent_blocks: 0,
                question_tokens: 0,
                applied_identity: None,
                state_forked: true,
                holds_parent_state: true,
            },
        );

        scheduler.discard_turn_view(view);
        assert_eq!(
            probe.get(parent),
            Some(held),
            "the parent has its state back"
        );
        assert_eq!(probe.get(view.0), None);
    }

    /// **A view forked at admission holds a copy, and its discard leaves the
    /// parent exactly where it stood**: the faulted turn's state goes with it.
    #[test]
    fn a_discarded_forked_view_leaves_the_parent_as_it_stood() {
        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
        let parent = scheduler.session.create_sequence().unwrap();
        forward_on(&mut scheduler, parent, &[1, 2, 3, 4]);
        let held = probe.get(parent).expect("the parent carries state");

        let (view, borrowed) = scheduler
            .create_view(SequenceId(parent), &[])
            .expect("view creation");
        forward_on(&mut scheduler, view.0, &[5, 6]);
        assert_ne!(probe.get(view.0), Some(held), "the view advanced its copy");
        scheduler.turn_views.insert(
            view,
            ViewState {
                parent_id: SequenceId(parent),
                original_borrowed: borrowed,
                turn_start_parent_blocks: 0,
                question_tokens: 0,
                applied_identity: None,
                state_forked: true,
                holds_parent_state: false,
            },
        );

        scheduler.discard_turn_view(view);
        assert_eq!(probe.get(parent), Some(held), "the parent is as it stood");
        assert_eq!(probe.get(view.0), None);
    }

    /// **A parked slot is not idle while anything runs on it**, and is once
    /// nothing does: a turn view carved from it keeps its store.
    #[test]
    fn a_slot_with_a_turn_carved_from_it_is_not_idle() {
        let (mut scheduler, _tx, _probe) = make_test_scheduler_recurrent();
        let parent = SequenceId(scheduler.session.create_sequence().unwrap());
        assert!(scheduler.slot_idle(parent));
        let (view, borrowed) = scheduler.carve_view(parent, &[]).expect("view carve");
        scheduler.turn_views.insert(
            view,
            ViewState {
                parent_id: parent,
                original_borrowed: borrowed,
                turn_start_parent_blocks: 0,
                question_tokens: 0,
                applied_identity: None,
                state_forked: true,
                holds_parent_state: false,
            },
        );
        assert!(!scheduler.slot_idle(parent), "a turn runs on it");
        scheduler.discard_turn_view(view);
        assert!(scheduler.slot_idle(parent));
    }

    /// **A parked slot whose record cannot be read back is an error**, never a
    /// slot that starts again from sequence-start: that would read fluently and
    /// remember nothing.
    #[test]
    fn a_parked_slot_without_a_readable_record_refuses_to_start_over() {
        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
        let slot = scheduler.session.create_sequence().unwrap();
        forward_on(&mut scheduler, slot, &[1, 2, 3, 4]);
        assert!(scheduler.model.park_recurrent(slot).unwrap());
        assert_eq!(probe.get(slot), None, "the store is gone");
        let refused = scheduler.unpark_from_record(SequenceId(slot));
        assert!(refused.is_err(), "no record: the slot does not start over");
        assert!(scheduler.model.recurrent_parked(slot), "and stays parked");
    }

    /// **An unparked slot passes straight through**: nothing to stage.
    #[test]
    fn an_unparked_slot_needs_no_record() {
        let (mut scheduler, _tx, _probe) = make_test_scheduler_recurrent();
        let slot = scheduler.session.create_sequence().unwrap();
        forward_on(&mut scheduler, slot, &[1, 2]);
        assert!(scheduler.unpark_from_record(SequenceId(slot)).is_ok());
    }

    /// The seeded flag suppresses **exactly one** reset.
    ///
    /// A forked view legitimately stands at offset 0 before its first wave —
    /// it has decoded nothing yet — and it holds state that was put there on
    /// purpose. Without the flag, `ensure_recurrent`'s reset fires and the
    /// fork was pointless. With a flag that never clears, a later genuine reset
    /// is suppressed and a recycled slot inherits a stranger's memory.
    #[test]
    fn the_seeded_flag_suppresses_exactly_one_reset() {
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };
        probe.ensure(0, 0);
        probe.set(0, [[7.0; 4]; 2]);
        model.fork_recurrent(0, 1).unwrap();

        // First wave at offset 0: the fork survives.
        probe.ensure(1, 0);
        assert_eq!(
            probe.get(1),
            Some([[7.0; 4]; 2]),
            "the reset wiped a freshly forked view before it ran a single token"
        );

        // Second wave at offset 0: this one is a genuine reset.
        probe.ensure(1, 0);
        assert_eq!(
            probe.get(1),
            Some(ZERO_STATE),
            "the seeded flag outlived its one wave and suppressed a real reset"
        );
    }

    /// **A resume that cannot install the state falls back to zeros — audibly.**
    ///
    /// Every rejection path leaves the slot at the sequence-start value, which
    /// is a conversation that comes back fluent and having forgotten. That is
    /// the failure this whole path exists to remove, so it must not *also* be
    /// the quiet path: the three reasons are distinguishable and each logs.
    ///
    /// Driven through `restore_recurrent_state` itself rather than asserted on
    /// the log, because what matters is that the slot ends at zeros and the
    /// scheduler carries on — a resume failure must never take the conversation
    /// down with it.
    #[test]
    fn a_refused_resume_leaves_the_slot_at_zeros_and_does_not_fail_the_open() {
        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
        let conversation = crate::projection::Conversation::new();
        let timeline = TimelineId::for_test(9);

        // No snapshot for this timeline at all — the ordinary case for a model
        // that carries no recurrent state, or a first turn that has not sealed.
        let slot = SequenceId(scheduler.session.create_sequence().unwrap());
        scheduler.restore_recurrent_state(slot, &conversation, timeline);
        assert_eq!(
            probe.get(slot.0),
            None,
            "a missing snapshot must leave the slot untouched, not half-written"
        );

        // And the slot still opens and runs: `ensure_recurrent` gives it the
        // sequence-start value, exactly as a fresh conversation gets.
        probe.ensure(slot.0, 0);
        assert_eq!(probe.get(slot.0), Some(ZERO_STATE));
        assert!(
            !probe.is_seeded(slot.0),
            "nothing was restored, so nothing may claim to be seeded — a stale \
             seeded mark would suppress this slot's next genuine reset"
        );
    }

    /// **One turn is priced at one store, whatever the engine is holding.**
    ///
    /// The price comes from the model's geometry, so nothing about the
    /// scheduler's occupancy can enter it. Both answers the old arithmetic gave
    /// are asserted against by name, because both were wrong in production and
    /// either would come back if someone reached for the total again:
    ///
    /// * `total / live` with nothing in flight divided by zero and answered
    ///   **0** — a 126 MiB claim priced as free, admitted, and refused by the
    ///   span on contact.
    /// * the same expression with one sequence in flight answered the **whole
    ///   total**, which is how a 41-row turn came to be priced at 4,450 MiB and
    ///   refused as throughput-worse with seven turns queued behind it.
    ///
    /// A `live.max(1)` patch is caught by the second assertion too — it returns
    /// the total here rather than one store.
    #[test]
    fn one_turn_is_priced_at_one_store_however_many_are_parked() {
        let (sched, _tx, _probe) = make_test_scheduler_recurrent();
        let store = DummyRecurrentModel::STORE_BYTES as u64;
        let total = DummyRecurrentModel::RESERVED_TOTAL as u64;

        assert_eq!(
            sched.recurrent_cost(),
            store,
            "one store, from the geometry"
        );
        assert_ne!(sched.recurrent_cost(), 0, "priced free with nothing live");
        assert_ne!(
            sched.recurrent_cost(),
            total,
            "billed for every store the process holds"
        );
    }

    /// **Speculative decode is refused up front on a model that cannot rewind.**
    ///
    /// The accept step puts the sequence back to the accepted prefix, which a
    /// recurrent state cannot express. The refusal has to happen at the entry
    /// point rather than inside the rewind: by the time the driver has drafted
    /// and verified, erroring is an error against work already done, and *not*
    /// erroring is a state that has absorbed tokens the K/V no longer holds.
    #[test]
    fn speculative_decode_is_refused_for_a_model_carrying_recurrent_state() {
        struct Recurrent(DummyRecurrentModel);
        impl ManagedBatchedModel for Recurrent {
            fn maybe_change_dtype(&self, d: DType) -> candle::Result<()> {
                self.0.maybe_change_dtype(d)
            }
            fn num_layers(&self) -> usize {
                self.0.num_layers()
            }
            fn n_kv_head(&self) -> usize {
                self.0.n_kv_head()
            }
            fn head_dim(&self) -> usize {
                self.0.head_dim()
            }
            fn wave_geometry(&self, d: DType) -> candle_nn::kv_cache::ModelGeometry {
                self.0.wave_geometry(d)
            }
            fn device(&self) -> &candle::Device {
                self.0.device()
            }
            #[allow(clippy::too_many_arguments)]
            fn forward_wave(
                &self,
                s: &mut BatchedInferenceSession,
                a: &[usize],
                b: &[Tensor],
                c: &[usize],
                d: &[Tensor],
                e: &[usize],
                f: &[Tensor],
                g: usize,
                h: usize,
                i: Option<&mut CreepResidual>,
            ) -> candle::Result<candle_transformers::models::batched_inference::WaveResult>
            {
                self.0.forward_wave(s, a, b, c, d, e, f, g, h, i)
            }
            fn forward_step(
                &self,
                session: &mut BatchedInferenceSession,
                spec: StepSpec<'_>,
            ) -> candle::Result<StepOutcome> {
                stub_step(self, session, spec)
            }
            fn claim_residual(
                &self,
                _session: &BatchedInferenceSession,
                rows: usize,
            ) -> candle::Result<CreepResidual> {
                stub_residual(rows)
            }
            fn prune(&self) -> candle::Result<()> {
                Ok(())
            }
            fn carries_recurrent_state(&self) -> bool {
                true
            }
        }

        let model = Recurrent(DummyRecurrentModel::new());
        let mut session = make_test_session();
        let seq = session.create_sequence().unwrap();
        let mut sink: Vec<Box<dyn FnMut(u32) -> bool + '_>> = vec![Box::new(|_| true)];
        // The chooser is immaterial here: the refusal is a property of the model
        // (recurrent state it cannot rewind), so it must fire before any row is
        // ever scored. Greedy is the one whose behaviour needs no explanation if
        // that ordering ever regresses and this reaches the sampling step.
        let mut chooser = GreedyChooser::whole_row();

        let err = match model.speculative_decode_step_batch(
            &mut session,
            &[seq],
            &[1u32],
            &[4],
            model.num_layers(),
            &mut chooser,
            &mut sink,
        ) {
            Ok(_) => panic!(
                "speculative decode ran on a model that cannot rewind — the accept \
                 step would leave the state holding rejected draft tokens"
            ),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("recurrent state"),
            "the refusal must name the reason, got: {err}"
        );
    }

    /// **F2's gate input: a model carrying recurrent state cannot gap-fill.**
    ///
    /// `reserve_glue_island` refuses to plan a mid-sequence glue island for a
    /// model whose `model_core_properties().can_gap_fill` is false, naming the
    /// island size and the reason — a recurrence's state has accumulated past
    /// the hole and cannot re-enter it. That refusal is only as good as this
    /// property, which the test double used to get wrong (see
    /// [`DummyRecurrentModel::carries_recurrent_state`]): a model that forks
    /// and moves state while reporting `can_gap_fill = true` sails past the
    /// refusal and fails a layer down, after the chunks are reserved.
    #[test]
    fn a_recurrent_model_reports_it_cannot_gap_fill() {
        let model = DummyRecurrentModel::new();
        assert!(model.carries_recurrent_state());
        assert!(
            !model.model_core_properties().can_gap_fill,
            "a model with recurrent per-sequence memory claimed it can compute \
             K/V for a token inserted mid-sequence — the projection would plan \
             glue islands the wave must then refuse"
        );
    }

    /// Forking from a parent that has never run a wave is not an error — a view
    /// is carved on a brand-new conversation's first turn, before the parent
    /// has any state. The child correctly begins at the sequence-start value.
    #[test]
    fn forking_a_stateless_parent_is_not_an_error() {
        let probe = RecurrentProbe::default();
        let model = DummyRecurrentModel {
            inner: DummyModel::new(),
            probe: probe.clone(),
        };
        model.fork_recurrent(0, 1).expect("a cold parent is legal");
        assert_eq!(probe.get(1), None, "nothing to copy, so nothing copied");
        probe.ensure(1, 0);
        assert_eq!(probe.get(1), Some(ZERO_STATE), "and it starts from zero");
    }

    /// **The wiring, not just the hook.** Drives `Scheduler::create_view`
    /// itself and asserts the view came out carrying the parent's state.
    ///
    /// The hook-level tests above would all pass with `create_view` never
    /// calling `fork_recurrent` at all — which is exactly the shape of the
    /// defect this phase fixes, so the wiring needs its own assertion.
    #[test]
    fn scheduler_create_view_forks_the_parents_recurrent_state() {
        let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();

        let parent_raw = scheduler.session.create_sequence().unwrap();
        let parent_id = SequenceId(parent_raw);
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let input = candle::Tensor::new(&tokens[..], &scheduler.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        scheduler
            .session
            .ensure_capacity(&[parent_raw], tokens.len())
            .unwrap();
        let nl = scheduler.model.num_layers().max(1);
        scheduler
            .model
            .forward_wave(
                &mut scheduler.session,
                &[],
                &[],
                &[parent_raw],
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
            .advance_sequence(parent_raw, tokens.len())
            .unwrap();

        // The parent's prefill advanced its state by the 8 prompt tokens.
        let mut expected = ZERO_STATE;
        toy_advance(&mut expected, tokens.len());
        assert_state_eq(&probe, parent_raw, expected);

        let (view_id, _) = scheduler
            .create_view(parent_id, &[BlockRange::new(0, 1)])
            .expect("view creation");

        assert_state_eq(&probe, view_id.0, expected);
        assert!(
            probe.is_seeded(view_id.0),
            "the forked view must be seeded, or its first wave at offset 0 \
             resets the state the fork just copied"
        );
    }
}
