//! A turn that ends on a fault is discarded, never sealed.
//!
//! A forward that has gone non-finite keeps producing token 0, and the sampler
//! stops the turn after a run of them. Ending it as an ordinary turn sealed what
//! it decoded — K/V over non-finite activations — into the conversation, where
//! every later turn attended over it and failed the same way. A stub whose
//! every decode row picks token 0 stands in for that forward.

use super::*;
use crate::batched_sampler::DEGENERATE_TOKEN_RUN;
use crate::guest::Guests;
use candle::{Device, Result};
use candle_nn::kv_cache::ModelGeometry;
use candle_transformers::models::batched_inference::WaveStep;

const VOCAB: usize = 512;
/// Not token 0, so a run of token 0 is a degenerate run rather than an EOS.
const EOS: u32 = VOCAB as u32 - 1;

/// A one-layer stub whose every decode row picks token 0.
#[derive(Clone)]
struct StuckOnZero {
    inner: DummyModel,
}

impl ManagedBatchedModel for StuckOnZero {
    fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
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
    fn wave_geometry(&self, act_dtype: DType) -> ModelGeometry {
        self.inner.wave_geometry(act_dtype)
    }
    fn device(&self) -> &Device {
        self.inner.device()
    }
    fn prune(&self) -> Result<()> {
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_wave(
        &self,
        _session: &mut BatchedInferenceSession,
        _decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        _prefill_seqs: &[usize],
        _prefill_inputs: &[Tensor],
        _glue_seqs: &[usize],
        _glue_inputs: &[Tensor],
        _layer_start: usize,
        _layer_end: usize,
        _residual_in: Option<Tensor>,
    ) -> Result<WaveResult> {
        let mut rows = Vec::new();
        for t in decode_inputs {
            for _ in t.flatten_all()?.to_vec1::<u32>()? {
                let mut row = vec![0f32; VOCAB];
                row[0] = 50.0;
                rows.push(Tensor::from_vec(row, (1, VOCAB), self.inner.device())?);
            }
        }
        Ok(WaveResult::owned(WaveStep {
            residual: None,
            logits: Some(rows),
        }))
    }
}

/// One decoding slot over the stuck stub.
fn stuck_slot() -> (Scheduler, SequenceId, Receiver<TurnEvent>) {
    let (_tx, rx) = flume::bounded(16);
    let mut scheduler = Scheduler::new(
        rx,
        Box::new(StuckOnZero {
            inner: DummyModel::new(),
        }),
        make_test_session(),
        make_dummy_tokenizer(),
        vec![EOS].into(),
        VOCAB,
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
    let id = SequenceId(scheduler.session.create_sequence().unwrap());
    let (mut state, events) = boundary_state();
    state.sampling_config = SamplingConfig::argmax();
    state.generated_tokens = TokenBuffer::from(vec![110]);
    scheduler.active_decodes.insert(id, state);
    scheduler
        .sampling_states
        .insert(id, SequenceSamplingState::new(VOCAB, 8));
    (scheduler, id, events)
}

/// What the turn's stream has carried so far: its errors, and whether it was
/// told the turn completed.
fn heard(events: &Receiver<TurnEvent>) -> (Vec<String>, bool) {
    let mut errors = Vec::new();
    let mut done = false;
    for e in events.try_iter() {
        match e {
            TurnEvent::Error(e) => errors.push(e.to_string()),
            TurnEvent::Done(_) => done = true,
            _ => {}
        }
    }
    (errors, done)
}

/// **A run of unusable logits fails the turn, and cleanup discards it.** The
/// caller hears one error naming the fault and is never told the turn
/// completed — completion is what sealed it.
#[test]
fn a_degenerate_run_fails_the_turn_and_it_is_never_sealed() {
    let (mut scheduler, slot, events) = stuck_slot();
    for step in 1..DEGENERATE_TOKEN_RUN {
        scheduler.batch_decode_step();
        assert!(
            !scheduler.active_decodes[&slot].faulted,
            "step {step} is still inside the bar"
        );
    }
    scheduler.batch_decode_step();
    let state = &scheduler.active_decodes[&slot];
    assert!(
        state.finished && state.faulted,
        "the run ends the turn on a fault"
    );

    scheduler.cleanup_finished();
    assert!(!scheduler.active_decodes.contains_key(&slot));
    let (errors, done) = heard(&events);
    assert_eq!(
        errors.len(),
        1,
        "one error, however the turn wound down: {errors:?}"
    );
    assert!(errors[0].contains("degenerate decode"), "{}", errors[0]);
    assert!(!done, "a faulted turn is never reported complete");
}

/// Every turn `fail_all_decodes` names is a faulted one, and a second failure
/// on the same turn is not a second error.
#[test]
fn failing_a_decode_twice_reports_once() {
    let (mut scheduler, slot, events) = stuck_slot();
    scheduler.fail_all_decodes(&[slot], "sampling failed: first");
    scheduler.fail_all_decodes(&[slot], "sampling failed: second");
    let state = &scheduler.active_decodes[&slot];
    assert!(state.finished && state.faulted);
    let (errors, done) = heard(&events);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("first"), "{}", errors[0]);
    assert!(!done);
}
