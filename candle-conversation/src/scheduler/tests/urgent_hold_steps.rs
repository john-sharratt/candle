//! Background decode rows held back while a person waits on a decode
//! (`urgent_hold`).
//!
//! A stub records which sequences each forward's decode slot carried, so each
//! test checks the scheduler's own choice of rows step by step.

use super::*;
use crate::guest::Guests;
use crate::projection::DecodePriority;
use crate::scheduler::urgent_hold::LOW_RIDES_EVERY;
use candle::{Device, Result};
use candle_nn::kv_cache::ModelGeometry;
use candle_transformers::models::batched_inference::WaveStep;

const VOCAB: usize = 512;
const EOS: u32 = VOCAB as u32 - 1;

/// The decode slot of every forward the stub ran, in order.
type Carried = Arc<Mutex<Vec<Vec<usize>>>>;

/// A one-layer stub whose every decode row picks the token after its input,
/// recording the sequences each forward's decode slot carried.
#[derive(Clone)]
struct Recorder {
    inner: DummyModel,
    carried: Carried,
}

impl ManagedBatchedModel for Recorder {
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
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        _prefill_seqs: &[usize],
        _prefill_inputs: &[Tensor],
        _glue_seqs: &[usize],
        _glue_inputs: &[Tensor],
        _layer_start: usize,
        _layer_end: usize,
        _residual: Option<&mut CreepResidual>,
    ) -> Result<WaveResult> {
        self.carried.lock().unwrap().push(decode_seqs.to_vec());
        let mut rows = Vec::new();
        for t in decode_inputs {
            for id in t.flatten_all()?.to_vec1::<u32>()? {
                let mut row = vec![0f32; VOCAB];
                row[(id as usize + 1) % (VOCAB - 1)] = 50.0;
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
}

/// A scheduler over the recorder, and what it recorded.
fn recording_scheduler() -> (Scheduler, Carried) {
    let carried: Carried = Arc::new(Mutex::new(Vec::new()));
    let (_tx, rx) = flume::bounded(16);
    let scheduler = Scheduler::new(
        rx,
        Box::new(Recorder {
            inner: DummyModel::new(),
            carried: Arc::clone(&carried),
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
    (scheduler, carried)
}

/// A decoding slot of `band` whose last committed token is `last`.
fn decoding(
    scheduler: &mut Scheduler,
    band: DecodePriority,
    last: u32,
) -> (SequenceId, Receiver<TurnEvent>) {
    let id = SequenceId(scheduler.session.create_sequence().unwrap());
    let (mut state, events) = boundary_state();
    state.sampling_config = SamplingConfig::argmax();
    state.generated_tokens = TokenBuffer::from(vec![last]);
    scheduler.active_decodes.insert(id, state);
    scheduler
        .sampling_states
        .insert(id, SequenceSamplingState::new(VOCAB, 8));
    scheduler.slot_bands.insert(id, band);
    (id, events)
}

/// **Beside a person's decode, the background rides one step in four.** The
/// `Urgent` row and the `High` one ride every step; the `Low` one rides the
/// fourth, and loses nothing while it waits — it decodes exactly the tokens it
/// would have, a step at a time.
#[test]
fn low_rows_ride_one_step_in_four_beside_an_urgent_decode() {
    let (mut scheduler, carried) = recording_scheduler();
    let (urgent, _u) = decoding(&mut scheduler, DecodePriority::Urgent, 10);
    let (high, _h) = decoding(&mut scheduler, DecodePriority::High, 100);
    let (low, _l) = decoding(&mut scheduler, DecodePriority::Low, 200);

    for _ in 0..2 * LOW_RIDES_EVERY {
        scheduler.batch_decode_step();
    }
    let carried = carried.lock().unwrap().clone();
    assert_eq!(
        carried.len(),
        2 * LOW_RIDES_EVERY as usize,
        "one forward a step"
    );
    for (step, rows) in carried.iter().enumerate() {
        assert!(rows.contains(&urgent.0), "step {step}: the dialogue rides");
        assert!(rows.contains(&high.0), "step {step}: High is not held");
        let low_rides = (step + 1) % LOW_RIDES_EVERY as usize == 0;
        assert_eq!(rows.contains(&low.0), low_rides, "step {step}: {rows:?}");
    }
    assert_eq!(
        &scheduler.active_decodes[&low].generated_tokens[..],
        &[200u32, 201, 202][..],
        "two steps ridden, two tokens, none skipped"
    );
    assert_eq!(scheduler.active_decodes[&urgent].generated_tokens.len(), 9);
}

/// **With nobody waiting, nothing is held.** `High` beside `Low` is the
/// cast beside ingest — the pairing the removed pause never let go of.
#[test]
fn without_an_urgent_decode_every_row_rides_every_step() {
    let (mut scheduler, carried) = recording_scheduler();
    let (high, _h) = decoding(&mut scheduler, DecodePriority::High, 100);
    let (low, _l) = decoding(&mut scheduler, DecodePriority::Low, 200);

    for _ in 0..LOW_RIDES_EVERY {
        scheduler.batch_decode_step();
    }
    let carried = carried.lock().unwrap().clone();
    assert_eq!(carried.len(), LOW_RIDES_EVERY as usize);
    for rows in &carried {
        assert!(rows.contains(&high.0) && rows.contains(&low.0), "{rows:?}");
    }
}

/// **A held row's tokens are not counted against a step's yield.** The
/// tokens a step commits are divided by the rows it carried, so a step that
/// held the background back reads as one token a row, not one in three.
#[test]
fn a_held_row_does_not_count_as_a_row_the_step_carried() {
    let (mut scheduler, _carried) = recording_scheduler();
    let (_urgent, _u) = decoding(&mut scheduler, DecodePriority::Urgent, 10);
    let (_low_a, _a) = decoding(&mut scheduler, DecodePriority::Low, 200);
    let (_low_b, _b) = decoding(&mut scheduler, DecodePriority::Low, 300);

    scheduler.batch_decode_step();
    assert_eq!(scheduler.step_rows, 1, "the dialogue alone rode");
}
