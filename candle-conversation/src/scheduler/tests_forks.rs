//! Tests for view forks at admission: a submitted turn's state is forked when
//! admission takes it, and a refused fork waits or fails the turn.
//!
//! A fork refused for want of span room waits while something is in flight. Its
//! view stays unforked, and it is forked on a later pass once there is room. With
//! nothing in flight, a refused fork fails the turn instead of waiting on ground
//! that nothing will free. A view that fails part way frees its slot, since a kept
//! slot would pin its parent's prefix with nothing left to release it. The session
//! refuses an empty range list, which would decode over empty K/V, while the
//! scheduler's sentinel for all blocks succeeds.

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

    use std::sync::Mutex;

    use candle_transformers::models::batched_inference::{
        BatchedInferenceSession, ManagedBatchedModel, WaveResult,
    };

    use candle_transformers::models::wave_driver::{StepOutcome, StepSpec};

    use super::super::tests_fixtures::tests::{
        make_dummy_tokenizer, make_test_scheduler_recurrent, make_test_session, stub_residual,
        stub_step, DummyModel,
    };

    use super::super::tests_admission::tests::test_prefill_work;

    /// A model that records every view fork and every staged snapshot, refuses
    /// every fork when `refuse` is set — a recurrent store the span has no room
    /// for — and refuses an eager restore outright, so a request that should
    /// stage and restores instead fails its test.
    #[derive(Clone)]
    pub(in super::super) struct ForkLog {
        pub(in super::super) inner: DummyModel,
        pub(in super::super) forks: Arc<Mutex<Vec<(usize, usize)>>>,
        pub(in super::super) staged: Arc<Mutex<Vec<usize>>>,
        pub(in super::super) refuse: bool,
    }

    impl ManagedBatchedModel for ForkLog {
        fn maybe_change_dtype(&self, dtype: DType) -> candle::Result<()> {
            self.inner.maybe_change_dtype(dtype)
        }
        fn num_layers(&self) -> usize {
            self.inner.num_layers()
        }
        fn n_kv_head(&self) -> usize {
            self.inner.n_kv_head()
        }
        fn head_dim(&self) -> usize {
            self.inner.head_dim()
        }
        fn wave_geometry(&self, act_dtype: DType) -> candle_nn::kv_cache::ModelGeometry {
            self.inner.wave_geometry(act_dtype)
        }
        fn device(&self) -> &candle::Device {
            self.inner.device()
        }
        #[allow(clippy::too_many_arguments)]
        fn forward_wave(
            &self,
            session: &mut BatchedInferenceSession,
            decode_seqs: &[usize],
            decode_inputs: &[Tensor],
            prefill_seqs: &[usize],
            prefill_inputs: &[Tensor],
            glue_seqs: &[usize],
            glue_inputs: &[Tensor],
            layer_start: usize,
            layer_end: usize,
            residual: Option<&mut CreepResidual>,
        ) -> candle::Result<WaveResult> {
            self.inner.forward_wave(
                session,
                decode_seqs,
                decode_inputs,
                prefill_seqs,
                prefill_inputs,
                glue_seqs,
                glue_inputs,
                layer_start,
                layer_end,
                residual,
            )
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
        fn fork_recurrent(&self, parent: usize, child: usize) -> candle::Result<()> {
            if self.refuse {
                candle::bail!("slot arena: no region for recurrent state")
            }
            self.forks.lock().unwrap().push((parent, child));
            Ok(())
        }
        fn restore_recurrent(
            &self,
            seq: usize,
            _schedule_hash: u64,
            _layers: &[ExportedLayerState],
        ) -> candle::Result<bool> {
            candle::bail!("sequence {seq} was restored eagerly instead of staged")
        }
        fn stage_recurrent(
            &self,
            seq: usize,
            _schedule_hash: u64,
            _layers: Arc<[ExportedLayerState]>,
        ) -> candle::Result<bool> {
            self.staged.lock().unwrap().push(seq);
            Ok(true)
        }
    }

    /// A scheduler over [`ForkLog`], and a handle on the logs it writes.
    pub(in super::super) fn fork_log_scheduler(refuse: bool) -> (Scheduler, ForkLog) {
        let model = ForkLog {
            inner: DummyModel::new(),
            forks: Arc::new(Mutex::new(Vec::new())),
            staged: Arc::new(Mutex::new(Vec::new())),
            refuse,
        };
        let log = model.clone();
        let (_tx, rx) = flume::bounded(16);
        let sched = Scheduler::new(
            rx,
            Box::new(model) as Box<dyn ManagedBatchedModel + Send>,
            make_test_session(),
            make_dummy_tokenizer(),
            vec![0u32].into(),
            64,
            8,
            false,
            None,
            DecodeHealthConfig::default(),
            512,
            PersistenceTrigger::noop(),
            SummariserTrigger::noop(),
            projection_assembler::BoundaryMarkers::default(),
            Arc::new(Guests::new()),
        );
        (sched, log)
    }

    /// Queue a turn the way `SubmitTurn` does: its view carved at submit, its
    /// state not yet forked.
    pub(in super::super) fn submit_unforked_turn(
        sched: &mut Scheduler,
    ) -> (SequenceId, SequenceId) {
        let parent = SequenceId(sched.session.create_sequence().unwrap());
        let (view, borrowed) = sched.carve_view(parent, &[]).unwrap();
        sched.turn_views.insert(
            view,
            ViewState {
                parent_id: parent,
                original_borrowed: borrowed,
                turn_start_parent_blocks: borrowed.0,
                question_tokens: 0,
                applied_identity: None,
                state_forked: false,
                holds_parent_state: false,
            },
        );
        sched.prefill_queue.push_back(test_prefill_work(view));
        (parent, view)
    }

    /// **A submitted turn's view is forked its parent's state when admission
    /// takes it, not when it is submitted.** A window of submissions forking at
    /// submit claimed a recurrent store for every turn at once, outside
    /// admission, and failed the ones the span had no room for.
    #[test]
    fn a_submitted_turns_state_is_forked_at_admission() {
        let (mut sched, log) = fork_log_scheduler(false);
        let (parent, view) = submit_unforked_turn(&mut sched);
        assert!(
            log.forks.lock().unwrap().is_empty(),
            "nothing forked at submit"
        );

        sched.promote_new_prefills();

        assert_eq!(*log.forks.lock().unwrap(), vec![(parent.0, view.0)]);
        assert!(sched.turn_views[&view].state_forked);
        assert_eq!(sched.active_prefills.len(), 1);
        assert!(sched.active_prefills[0].error.is_none());
    }

    /// **A refused fork waits while something is in flight**, its view kept and
    /// still unforked, so the next pass forks it once there is room.
    #[test]
    fn a_refused_fork_waits_while_something_is_in_flight() {
        let (mut sched, _) = fork_log_scheduler(true);
        let running = SequenceId(sched.session.create_sequence().unwrap());
        sched.begin_prefill(test_prefill_work(running));
        let (_, view) = submit_unforked_turn(&mut sched);

        sched.promote_new_prefills();

        assert_eq!(sched.prefill_queue.len(), 1, "back in the queue");
        assert_eq!(sched.prefill_queue[0].sequence_id, view);
        assert!(!sched.turn_views[&view].state_forked);
        assert_eq!(
            sched.active_prefills.len(),
            1,
            "only the one already running"
        );
    }

    /// With nothing in flight a refused fork fails the turn rather than
    /// waiting on ground nothing will free.
    #[test]
    fn a_refused_fork_with_nothing_in_flight_fails_the_turn() {
        let (mut sched, _) = fork_log_scheduler(true);
        submit_unforked_turn(&mut sched);

        sched.promote_new_prefills();

        assert!(sched.prefill_queue.is_empty(), "not retried forever");
        assert_eq!(sched.active_prefills.len(), 1);
        assert!(sched.active_prefills[0].error.is_some());
    }

    // —— view-creation tests —————————————————————————————————————————————————

    /// **A view that fails part way leaves no slot behind.**
    ///
    /// A slot kept would hold whatever it had borrowed with nothing to free it:
    /// on Qwen3.8-Flash-Next's calibration every refused carve would pin its
    /// parent's prefix, and compaction would report the borrowed history as
    /// writable — 66,471 times in a minute.
    #[test]
    fn a_view_that_fails_part_way_frees_its_slot() {
        let mut session = make_test_session();
        let parent = session.create_sequence().unwrap();
        let next_free = session.create_sequence().unwrap();
        session.free_sequence(next_free).unwrap();
        // The parent holds no blocks, so the borrow refuses a range past them.
        assert!(session.create_view_sequence(parent, &[(0, 3)]).is_err());
        assert_eq!(
            session.create_sequence().unwrap(),
            next_free,
            "the failed view's slot is free again",
        );
    }

    /// Explicit `visible_block_ranges` over a populated parent must create a valid view.
    #[test]
    fn create_view_with_explicit_ranges_creates_view() {
        let model = DummyModel::new();
        let (_tx, rx) = flume::bounded(16);
        let model_box = Box::new(model) as Box<dyn ManagedBatchedModel + Send>;
        let session = make_test_session();
        let tokenizer = make_dummy_tokenizer();
        let mut scheduler = Scheduler::new(
            rx,
            model_box,
            session,
            tokenizer,
            vec![0u32].into(),
            64,
            8,
            false,
            None,
            DecodeHealthConfig::default(),
            512,
            PersistenceTrigger::noop(),
            SummariserTrigger::noop(),
            projection_assembler::BoundaryMarkers::default(),
            // No guests: these exercise the wave loop, and an empty registry is
            // what every deployment that has not configured one carries.
            Arc::new(Guests::new()),
        );

        let parent_raw = scheduler.session.create_sequence().unwrap();
        let parent_id = SequenceId(parent_raw);
        // Populate one full chunk on the parent so the view has a block to borrow.
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

        let result = scheduler.create_view(parent_id, &[BlockRange::new(0, 1)]);
        assert!(result.is_ok(), "view creation failed: {:?}", result.err());
    }

    /// Sentinel (empty) ranges + zero-block parent yields a
    /// zero-block view (borrowing zero blocks is valid).
    #[test]
    fn create_view_sentinel_with_zero_block_parent_yields_empty_view() {
        let model = DummyModel::new();
        let (_tx, rx) = flume::bounded(16);
        let model_box = Box::new(model) as Box<dyn ManagedBatchedModel + Send>;
        let session = make_test_session();
        let tokenizer = make_dummy_tokenizer();
        let mut scheduler = Scheduler::new(
            rx,
            model_box,
            session,
            tokenizer,
            vec![0u32].into(),
            64,
            8,
            false,
            None,
            DecodeHealthConfig::default(),
            512,
            PersistenceTrigger::noop(),
            SummariserTrigger::noop(),
            projection_assembler::BoundaryMarkers::default(),
            // No guests: these exercise the wave loop, and an empty registry is
            // what every deployment that has not configured one carries.
            Arc::new(Guests::new()),
        );

        let parent_raw = scheduler.session.create_sequence().unwrap();
        let parent_id = SequenceId(parent_raw);

        let result = scheduler.create_view(parent_id, &[]);
        assert!(
            result.is_ok(),
            "sentinel + empty parent should succeed: {:?}",
            result.err()
        );
    }

    /// **The session refuses a view with no ranges; the scheduler's sentinel is
    /// what expands to "all blocks".**
    ///
    /// The two spellings look identical at a call site and mean opposite things.
    /// `Scheduler::create_view(&[])` is the sentinel and must succeed;
    /// `BatchedInferenceSession::create_view_sequence(&[])` names zero blocks to
    /// borrow and must not, because the view it used to return decoded at its
    /// parent's position over an **empty** K/V — fluent, plausible text with no
    /// relation to the prompt, and no error anywhere. That cost two tier-2 runs,
    /// each reported as a fork defect.
    #[test]
    fn a_session_view_with_no_ranges_is_refused_while_the_scheduler_sentinel_is_not() {
        let (mut scheduler, _tx, _probe) = make_test_scheduler_recurrent();
        let parent = scheduler.session.create_sequence().unwrap();
        // A real forward, so the parent owns a block and "all" is
        // distinguishable from "none". Advancing the offset alone allocates
        // nothing.
        let tokens = [1u32, 2, 3, 4];
        let input = Tensor::new(&tokens[..], &scheduler.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        scheduler
            .session
            .ensure_capacity(&[parent], tokens.len())
            .unwrap();
        let nl = scheduler.model.num_layers().max(1);
        scheduler
            .model
            .forward_wave(
                &mut scheduler.session,
                &[],
                &[],
                &[parent],
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
            .advance_sequence(parent, tokens.len())
            .unwrap();

        let err = scheduler
            .session
            .create_view_sequence(parent, &[])
            .expect_err("an empty range list must not silently borrow nothing");
        assert!(
            err.to_string().contains("no visible block ranges"),
            "the refusal must name the fix: {err}"
        );

        // A zero-LENGTH range is a different thing and stays legal: it is what
        // the scheduler's sentinel expands to for a parent with no blocks.
        assert!(
            scheduler
                .session
                .create_view_sequence(parent, &[(0, 0)])
                .is_ok(),
            "a zero-length range borrows nothing legitimately"
        );

        // And the sentinel itself still means "all blocks".
        let (view, borrowed) = scheduler
            .create_view(SequenceId(parent), &[])
            .expect("the scheduler sentinel expands to the parent's blocks");
        assert!(
            borrowed.0 > 0,
            "the sentinel borrowed {} blocks from a parent that has some",
            borrowed.0
        );
        let _ = view;
    }
}
