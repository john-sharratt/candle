//! A decode step whose transient tier the partition refuses.
//!
//! A refusal is placement, not computation: the step commits nothing, and the
//! ground it was short of comes back a step or two later — relief, a
//! compaction, or the weight side conceding between waves. So the turn keeps
//! its place and the step runs again; only a run of refusals long enough to
//! give up ends it. A stub drafter refuses its tier a set number of times, then
//! proposes nothing, so every step after that is a plain decode row predicting
//! `t + 1`.

use super::*;
use crate::guest::Guests;
use crate::scheduler::wave_retry::GIVE_UP_AFTER;
use candle::{Device, Result};
use candle_nn::kv_cache::{ModelGeometry, TIER_REFUSED_MARKER, TIER_SHORT_BY};
use candle_transformers::models::batched_inference::WaveStep;
use std::sync::atomic::{AtomicUsize, Ordering};

const VOCAB: usize = 512;

/// A one-layer stub whose every decode row predicts `t + 1`, and whose drafter
/// refuses its tier while `refusals` is above zero.
#[derive(Clone)]
struct RefusingDrafter {
    inner: DummyModel,
    refusals: Arc<AtomicUsize>,
}

impl ManagedBatchedModel for RefusingDrafter {
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
        _residual: Option<&mut CreepResidual>,
    ) -> Result<WaveResult> {
        let mut rows = Vec::new();
        for t in decode_inputs {
            for id in t.flatten_all()?.to_vec1::<u32>()? {
                let mut row = vec![0f32; VOCAB];
                row[id as usize + 1] = 50.0;
                rows.push(Tensor::from_vec(row, (1, VOCAB), self.inner.device())?);
            }
        }
        Ok(WaveResult::owned(WaveStep { logits: Some(rows) }))
    }

    fn forward_step(
        &self,
        session: &mut BatchedInferenceSession,
        spec: StepSpec<'_>,
    ) -> Result<StepOutcome> {
        stub_step(self, session, spec)
    }

    fn claim_residual(
        &self,
        _session: &BatchedInferenceSession,
        rows: usize,
    ) -> Result<CreepResidual> {
        stub_residual(rows)
    }

    fn speculative_draft(
        &self,
        _session: &mut BatchedInferenceSession,
        seqs: &[usize],
        _committed: &[u32],
        _max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let left = self.refusals.load(Ordering::Relaxed);
        if left > 0 {
            self.refusals.store(left - 1, Ordering::Relaxed);
            candle::bail!(
                "{TIER_REFUSED_MARKER}: {TIER_SHORT_BY}117440512 B — its transient tier needs \
                 956301312 B below the weight floor"
            );
        }
        Ok(vec![Vec::new(); seqs.len()])
    }
}

/// One decoding slot whose last committed token is 110, over a drafter that
/// refuses its tier `refusals` times.
fn refused_slot(refusals: usize) -> (Scheduler, SequenceId, Receiver<TurnEvent>) {
    let model = RefusingDrafter {
        inner: DummyModel::new(),
        refusals: Arc::new(AtomicUsize::new(refusals)),
    };
    let (_tx, rx) = flume::bounded(16);
    let mut scheduler = Scheduler::new(
        rx,
        Box::new(model),
        make_test_session(),
        make_dummy_tokenizer(),
        vec![0u32].into(),
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

/// The errors the turn's stream has carried so far.
fn errors(events: &Receiver<TurnEvent>) -> Vec<String> {
    events
        .try_iter()
        .filter_map(|e| match e {
            TurnEvent::Error(e) => Some(e.to_string()),
            _ => None,
        })
        .collect()
}

/// **A refused tier leaves the turn where it stood, and the next step runs
/// it.** On the live daemon a draft walk 117 MB short ended a reply 317
/// tokens in, over ground that came back moments later.
#[test]
fn a_refused_decode_step_keeps_the_turn_and_runs_again() {
    let (mut scheduler, slot, events) = refused_slot(3);
    for attempt in 1..=3 {
        scheduler.batch_decode_step();
        let state = &scheduler.active_decodes[&slot];
        assert!(!state.finished, "refusal {attempt} does not end the turn");
        assert_eq!(&state.generated_tokens[..], &[110u32][..]);
        assert_eq!(scheduler.session.sequence_offset(slot.0), Some(0));
        assert!(
            scheduler.decode_retry.take_wait().is_some(),
            "refusal {attempt} owes the loop a wait"
        );
    }
    scheduler.batch_decode_step();
    let state = &scheduler.active_decodes[&slot];
    assert_eq!(
        &state.generated_tokens[..],
        &[110u32, 111][..],
        "the step after the refusals commits as if none had happened"
    );
    assert_eq!(scheduler.session.sequence_offset(slot.0), Some(1));
    assert!(errors(&events).is_empty(), "the turn heard no error");
    assert_eq!(
        scheduler.decode_retry.take_wait(),
        None,
        "a step that ran owes no wait"
    );
}

/// **A tier refused on every attempt still ends the turn, with the error**,
/// rather than standing on ground that never comes back.
#[test]
fn a_tier_refused_on_every_attempt_fails_the_turn() {
    let (mut scheduler, slot, events) = refused_slot(GIVE_UP_AFTER as usize);
    for _ in 1..GIVE_UP_AFTER {
        scheduler.batch_decode_step();
    }
    assert!(!scheduler.active_decodes[&slot].finished);
    assert!(errors(&events).is_empty());

    scheduler.batch_decode_step();
    assert!(scheduler.active_decodes[&slot].finished);
    let errs = errors(&events);
    assert_eq!(errs.len(), 1);
    assert!(
        errs[0].contains(TIER_REFUSED_MARKER),
        "the turn hears what refused it: {}",
        errs[0]
    );
}
