//! Tests for the wave loop's failure handling: a failed wave fails only the turns
//! it carried, and adapters are batched apart.
//!
//! Closing tails on different adapters ride separate waves, because one wave
//! cannot carry two adapters, and a failed group reports only its own turns. A
//! sequence named twice in one batch forwards its spans in order after the batch
//! rather than twice inside one wave. A failed glue fill forgets the pieces its
//! assembly placed, so that a later rebuild does not keep zero chunks as though
//! they had been filled.

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
        make_dummy_tokenizer, make_test_session, stub_residual, stub_step, DummyModel,
    };

    /// A model whose every wave forward fails — what a glue fill can meet.
    #[derive(Clone)]
    pub(in super::super) struct FailingWave(pub(in super::super) DummyModel);

    impl ManagedBatchedModel for FailingWave {
        fn maybe_change_dtype(&self, dtype: DType) -> candle::Result<()> {
            self.0.maybe_change_dtype(dtype)
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
        fn wave_geometry(&self, act_dtype: DType) -> candle_nn::kv_cache::ModelGeometry {
            self.0.wave_geometry(act_dtype)
        }
        fn device(&self) -> &candle::Device {
            self.0.device()
        }
        #[allow(clippy::too_many_arguments)]
        fn forward_wave(
            &self,
            _session: &mut BatchedInferenceSession,
            _decode_seqs: &[usize],
            _decode_inputs: &[Tensor],
            _prefill_seqs: &[usize],
            _prefill_inputs: &[Tensor],
            _glue_seqs: &[usize],
            _glue_inputs: &[Tensor],
            _layer_start: usize,
            _layer_end: usize,
            _residual: Option<&mut CreepResidual>,
        ) -> candle::Result<WaveResult> {
            candle::bail!("wave forward failed")
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
    }

    /// A model that refuses a wave mixing adapters — as the real session does —
    /// records every prefill wave's sequences, and fails any wave carrying
    /// `fail_with`.
    #[derive(Clone)]
    pub(in super::super) struct AdapterStrict {
        pub(in super::super) inner: DummyModel,
        pub(in super::super) waves: Arc<Mutex<Vec<Vec<usize>>>>,
        pub(in super::super) fail_with: Option<usize>,
    }

    impl ManagedBatchedModel for AdapterStrict {
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
            session.wave_adapter(prefill_seqs)?;
            if self.fail_with.is_some_and(|s| prefill_seqs.contains(&s)) {
                candle::bail!("wave forward failed");
            }
            self.waves.lock().unwrap().push(prefill_seqs.to_vec());
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
    }

    pub(in super::super) fn scheduler_over(model: AdapterStrict) -> Scheduler {
        let (_tx, rx) = flume::bounded(16);
        Scheduler::new(
            rx,
            Box::new(model),
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
        )
    }

    /// **Closing tails batch by adapter.** One wave cannot carry two adapters, so
    /// a turn on another adapter rides a wave of its own — and every tail lands.
    #[test]
    fn closing_tails_on_different_adapters_ride_separate_waves() {
        let waves = Arc::new(Mutex::new(Vec::new()));
        let mut sched = scheduler_over(AdapterStrict {
            inner: DummyModel::new(),
            waves: Arc::clone(&waves),
            fail_with: None,
        });
        let a = SequenceId(sched.session.create_sequence().unwrap());
        let b = SequenceId(sched.session.create_sequence().unwrap());
        let c = SequenceId(sched.session.create_sequence().unwrap());
        sched
            .session
            .set_sequence_adapter(b.0, Some(Arc::from("tuned")))
            .unwrap();
        let failed = sched.run_prefill_batch(&[(a, &[1, 2]), (b, &[3]), (c, &[4])]);
        assert!(failed.is_empty(), "{failed:?}");
        assert_eq!(*waves.lock().unwrap(), vec![vec![a.0, c.0], vec![b.0]]);
        assert_eq!(sched.session.sequence_offset(a.0), Some(2));
        assert_eq!(sched.session.sequence_offset(b.0), Some(1));
        assert_eq!(sched.session.sequence_offset(c.0), Some(1));
    }

    /// **A failed wave fails the turns it carried and no others.** The adapter
    /// group whose wave fails reports its turns; the other group's tails still
    /// land.
    #[test]
    fn a_failed_closing_tail_wave_fails_only_the_turns_it_carried() {
        let waves = Arc::new(Mutex::new(Vec::new()));
        let mut sched = scheduler_over(AdapterStrict {
            inner: DummyModel::new(),
            waves: Arc::clone(&waves),
            fail_with: None,
        });
        let a = SequenceId(sched.session.create_sequence().unwrap());
        let b = SequenceId(sched.session.create_sequence().unwrap());
        let c = SequenceId(sched.session.create_sequence().unwrap());
        sched
            .session
            .set_sequence_adapter(b.0, Some(Arc::from("tuned")))
            .unwrap();
        // Rebuild over a model that fails the wave carrying `a`.
        sched.model = Box::new(AdapterStrict {
            inner: DummyModel::new(),
            waves: Arc::clone(&waves),
            fail_with: Some(a.0),
        });
        let failed = sched.run_prefill_batch(&[(a, &[1, 2]), (b, &[3]), (c, &[4])]);
        let failed_ids: Vec<SequenceId> = failed.iter().map(|(s, _)| *s).collect();
        assert_eq!(failed_ids, vec![a, c], "the wave carrying a also carried c");
        assert_eq!(*waves.lock().unwrap(), vec![vec![b.0]]);
        assert_eq!(sched.session.sequence_offset(b.0), Some(1));
        assert_eq!(sched.session.sequence_offset(a.0), Some(0));
    }

    /// **A sequence named twice keeps its spans in order.** One wave cannot carry
    /// it twice, so the repeat forwards after the batch, on its own.
    #[test]
    fn a_sequence_named_twice_forwards_its_spans_in_order() {
        let waves = Arc::new(Mutex::new(Vec::new()));
        let mut sched = scheduler_over(AdapterStrict {
            inner: DummyModel::new(),
            waves: Arc::clone(&waves),
            fail_with: None,
        });
        let a = SequenceId(sched.session.create_sequence().unwrap());
        let b = SequenceId(sched.session.create_sequence().unwrap());
        let failed = sched.run_prefill_batch(&[(a, &[1]), (b, &[2]), (a, &[3, 4])]);
        assert!(failed.is_empty(), "{failed:?}");
        assert_eq!(*waves.lock().unwrap(), vec![vec![a.0, b.0], vec![a.0]]);
        assert_eq!(sched.session.sequence_offset(a.0), Some(3));
    }

    /// **A glue fill that fails leaves nothing of its assembly to keep.** The
    /// walk records each glue piece as placed before the fill computes it; a
    /// deferred fill taken off the queue and lost with its wave left those
    /// pieces as zero chunks, and every later rebuild kept them because its
    /// identities still matched. Losing the record costs one full rebuild.
    #[test]
    fn a_failed_glue_fill_forgets_the_pieces_its_assembly_placed() {
        use projection_assembler::{GapFillPlan, PlacedPiece};

        let (_tx, rx) = flume::bounded(16);
        let mut sched = Scheduler::new(
            rx,
            Box::new(FailingWave(DummyModel::new())),
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
        let slot = SequenceId(3);
        sched
            .slot_projection_state
            .entry(slot)
            .or_default()
            .placed_pieces
            .push(PlacedPiece::ending_at(2));
        sched.deferred_glue_fires.push(GapFillPlan {
            parent_id: slot,
            glue_tokens: vec![5, 6],
            glue_write_slice: vec![2, 2],
            glue_write_in_blk: vec![0, 1],
            fwd_ahead: vec![0, 0],
            deferred_user: None,
            tail_per_layer: Vec::new(),
            held_before: None,
            placed_before: 0,
            n_glue_tokens: 2,
            islands: Vec::new(),
            complete: true,
        });

        assert!(sched
            .decode_forward_cobatched(&[], &[], &[], &[], &[], &[])
            .is_err());
        assert!(
            sched.deferred_glue_fires.is_empty(),
            "the wave took the glue"
        );
        assert!(
            sched.slot_projection_state[&slot].placed_pieces.is_empty(),
            "a rebuild must not keep glue that was never filled"
        );
    }
}
