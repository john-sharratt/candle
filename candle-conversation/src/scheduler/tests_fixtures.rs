//! Shared fixtures for the scheduler tests: a dummy model, a CPU-backed session,
//! and a scheduler built over them without a GPU.
//!
//! The dummy model answers the scheduler's shape questions and runs no kernels,
//! and its shape is coherent so that anything priced against it gets a coherent
//! answer. The session has one layer and one head of width sixteen, which is
//! enough for the handler-level state the tests drive. The helpers create slots
//! through the ordinary creation handler, so every sequence under test is a real
//! session slot rather than an invented id. The `forward_on` helper advances a
//! slot the way a prefill unit does, and the boundary tests rely on that.

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

    use candle_transformers::models::batched_inference::{
        BatchedConfig, BatchedInferenceSession, KvLayers, ManagedBatchedModel,
    };

    use candle_transformers::models::wave_driver::{StepOutcome, StepSpec};

    use std::str::FromStr;

    use super::super::tests_toy_model::tests::{DummyRecurrentModel, RecurrentProbe};

    // —— Dummy model ——————————————————————————————————————————————————————————

    #[derive(Clone)]
    pub(in super::super) struct DummyModel {
        pub(in super::super) device: candle::Device,
        pub(in super::super) vocab_size: usize,
    }

    impl DummyModel {
        pub(in super::super) fn new() -> Self {
            Self {
                device: candle::Device::Cpu,
                vocab_size: 64,
            }
        }

        pub(in super::super) fn dummy_logits(&self, n: usize) -> candle::Result<Vec<Tensor>> {
            (0..n)
                .map(|_| Tensor::zeros((1, self.vocab_size), DType::F32, &self.device))
                .collect()
        }
    }

    impl ManagedBatchedModel for DummyModel {
        /// No norms to materialise: this stub answers the scheduler's shape
        /// questions and never runs a kernel.
        fn maybe_change_dtype(&self, _dtype: DType) -> candle::Result<()> {
            Ok(())
        }

        fn num_layers(&self) -> usize {
            1
        }
        fn n_kv_head(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            16
        }

        /// Matches the single-head, 16-wide shape the rest of this dummy
        /// reports, so anything that prices a wave against it gets a coherent
        /// (if tiny) answer rather than a placeholder.
        fn wave_geometry(&self, act_dtype: DType) -> candle_nn::kv_cache::ModelGeometry {
            candle_nn::kv_cache::ModelGeometry {
                hidden: 16,
                intermediate: 32,
                n_head: 1,
                n_kv_head: 1,
                head_dim: 16,
                experts_per_tok: 1,
                n_experts: 1,
                act_dtype,
                accum_dtype: DType::F32,
                vocab: 64,
                delta_net: None,
                shared_expert: None,
                packed_norm: false,
                packed_head: false,
                gated_qkv: false,
                fused_qkv: false,
                qkv_bias: false,
                head_qk_norm: false,
                head_norm_reshapes: false,
                partial_rotary: false,
                decode_q8_context: false,
                hyper: None,
                mtp_head: false,
            }
        }
        fn device(&self) -> &candle::Device {
            &self.device
        }

        #[allow(clippy::too_many_arguments)]
        fn forward_wave(
            &self,
            _session: &mut BatchedInferenceSession,
            decode_seqs: &[usize],
            _decode_inputs: &[Tensor],
            prefill_seqs: &[usize],
            _prefill_inputs: &[Tensor],
            _glue_seqs: &[usize],
            _glue_inputs: &[Tensor],
            _layer_start: usize,
            layer_end: usize,
            _residual: Option<&mut CreepResidual>,
        ) -> candle::Result<candle_transformers::models::batched_inference::WaveResult> {
            use candle_transformers::models::batched_inference::{WaveResult, WaveStep};
            let logits = if layer_end >= self.num_layers() {
                Some(self.dummy_logits(decode_seqs.len() + prefill_seqs.len())?)
            } else {
                None
            };
            Ok(WaveResult::owned(WaveStep { logits }))
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

    /// A one-layer stub's step: every row rides the stub's single forward, so a
    /// creep window — which on one layer always ends at the head — reaches the
    /// head with the rest, and its arena goes with it.
    pub(in super::super) fn stub_step<M: ManagedBatchedModel + ?Sized>(
        model: &M,
        session: &mut BatchedInferenceSession,
        spec: StepSpec<'_>,
    ) -> candle::Result<StepOutcome> {
        let n = model.num_layers();
        let StepSpec {
            decode,
            verify,
            glue,
            creep,
        } = spec;
        let (creep_seqs, creep_inputs): (&[usize], &[Tensor]) = match &creep {
            Some(c) => (c.rows.seqs, c.rows.inputs),
            None => (&[], &[]),
        };
        if let Some(c) = &creep {
            if c.win_end < n {
                candle::bail!(
                    "stub step: a creep window [{}, {}) short of {n} layers",
                    c.cursor,
                    c.win_end
                );
            }
        }
        let pre_seqs: Vec<usize> = verify.seqs.iter().chain(creep_seqs).copied().collect();
        let pre_inputs: Vec<Tensor> = verify.inputs.iter().chain(creep_inputs).cloned().collect();
        let head = model.forward_wave(
            session,
            decode.seqs,
            decode.inputs,
            &pre_seqs,
            &pre_inputs,
            glue.seqs,
            glue.inputs,
            0,
            n,
            None,
        );
        let Some(c) = creep else {
            return Ok(StepOutcome {
                creep_advanced: false,
                creep_us: 0,
                head,
            });
        };
        // The creep rode the one forward, so its failure is the creep's.
        let head = head?;
        *c.residual = None;
        Ok(StepOutcome {
            creep_advanced: true,
            creep_us: 0,
            head: Ok(head),
        })
    }

    /// A stub's residual arena: one-wide rows on the host.
    pub(in super::super) fn stub_residual(rows: usize) -> candle::Result<CreepResidual> {
        CreepResidual::claim(&candle::Device::Cpu, rows, 1, DType::F32)
    }

    // —— Helpers ——————————————————————————————————————————————————————————————

    /// Minimal CPU-backed session: 1 layer, 1 KV head, head_dim=16.
    pub(in super::super) fn make_test_session() -> BatchedInferenceSession {
        BatchedInferenceSession::new(
            KvLayers::stream_only(1),
            1,  // n_kv_head
            16, // head_dim
            &candle::Device::Cpu,
            BatchedConfig::default()
                .with_initial_seq_len(32)
                .with_dtype(DType::F32),
        )
        .expect("failed to create test BatchedInferenceSession on CPU")
    }

    /// Trivial WordLevel tokenizer with an empty vocab.
    /// The scheduler dispatch tests do not call encode/decode, so any valid
    /// Tokenizer object suffices here.
    pub(in super::super) fn make_dummy_tokenizer() -> tokenizers::Tokenizer {
        tokenizers::Tokenizer::from_str(
            r#"{"version":"1.0","truncation":null,"padding":null,
               "added_tokens":[],"normalizer":null,"pre_tokenizer":null,
               "post_processor":null,"decoder":null,
               "model":{"type":"WordLevel","vocab":{},"unk_token":"[UNK]"}}"#,
        )
        .expect("failed to build dummy tokenizer")
    }

    /// A scheduler over the CPU test session and a `DummyModel`, plus its
    /// request sender — for tests that drive handler-level state (belief
    /// lifecycle) rather than forwards.
    pub(in super::super) fn make_test_scheduler() -> (Scheduler, Sender<SchedulerRequest>) {
        let (tx, rx) = flume::bounded(16);
        let session = make_test_session();
        let tokenizer = make_dummy_tokenizer();
        let scheduler = Scheduler::new(
            rx,
            Box::new(DummyModel::new()),
            session,
            tokenizer,
            vec![0u32].into(), // eos_tokens
            64,                // vocab_size
            8,                 // max_recent_len
            false,             // show_special_tokens
            None,              // penalty_log_path
            DecodeHealthConfig::default(),
            512, // max_prefill_pass_tokens
            PersistenceTrigger::noop(),
            SummariserTrigger::noop(),
            projection_assembler::BoundaryMarkers::default(),
            // No guests: these exercise the wave loop, and an empty registry is
            // what every deployment that has not configured one carries.
            Arc::new(Guests::new()),
        );
        (scheduler, tx)
    }

    /// [`make_test_scheduler`] over a [`DummyRecurrentModel`], returning the
    /// probe as well so a test can read the state the scheduler's boxed model
    /// is carrying.
    pub(in super::super) fn make_test_scheduler_recurrent(
    ) -> (Scheduler, Sender<SchedulerRequest>, RecurrentProbe) {
        let (tx, rx) = flume::bounded(16);
        let session = make_test_session();
        let tokenizer = make_dummy_tokenizer();
        let model = DummyRecurrentModel::new();
        let probe = model.probe.clone();
        let scheduler = Scheduler::new(
            rx,
            Box::new(model),
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
        (scheduler, tx, probe)
    }

    // —— unit-boundary tests ————————————————————————————————————————————————
    //
    // The K/V anchor and the index boundary are one fact, and these pin the one
    // place that records it. Every one of them would have failed against the
    // three earlier placements — a cut in `apply_segments_finish` (a path the
    // ingest turn never takes), a cut in `prepare_section_ingest` alone (ditto),
    // and a cut at prefill admission (right in effect, but on the view *after*
    // the fork, so the parent's tail stayed open and grew across turns).

    /// Forward `tokens` on `slot` and advance it, the way a prefill unit does.
    pub(in super::super) fn forward_on(scheduler: &mut Scheduler, slot: usize, tokens: &[u32]) {
        let input = candle::Tensor::new(tokens, &scheduler.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        scheduler
            .session
            .ensure_capacity(&[slot], tokens.len())
            .unwrap();
        let nl = scheduler.model.num_layers().max(1);
        scheduler
            .model
            .forward_wave(
                &mut scheduler.session,
                &[],
                &[],
                &[slot],
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
            .advance_sequence(slot, tokens.len())
            .unwrap();
    }

    /// A scratch slot allocated through the ordinary handler, so the test's
    /// sequences are real session slots rather than invented ids.
    pub(in super::super) fn create_scratch_slot(
        scheduler: &mut Scheduler,
        conversation: &Conversation,
    ) -> SequenceId {
        let (tx, rx) = flume::bounded(1);
        scheduler.handle_request(SchedulerRequest::NewSequence {
            conversation: conversation.clone(),
            target: None,
            parent: None,
            response_tx: tx,
        });
        rx.recv().expect("scheduler reply").expect("slot allocated")
    }
}
