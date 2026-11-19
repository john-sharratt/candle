//! A creep cohort crawling beside a person's decode tells its members it moves.
//!
//! Beside an `Urgent` decode a `Low` cohort rides a layer a window
//! (`throttle_ratio`), and a chunk lands only once it has run every layer. A
//! caller's stall clock that heard nothing between landings gave such a turn up
//! as never admitted: every file read beside a long reply on BlackWorld
//! (2026-11-19). A four-layer stub runs the windows here, so each test checks
//! what the scheduler tells a member at each one.

use super::*;
use crate::guest::Guests;
use crate::projection::DecodePriority;
use candle::{Device, Result};
use candle_nn::kv_cache::ModelGeometry;
use candle_transformers::models::batched_inference::WaveStep;

const VOCAB: usize = 512;
const EOS: u32 = VOCAB as u32 - 1;
const LAYERS: usize = 4;

/// A four-layer stub: every decode row picks the token after its input, every
/// creep member's last row scores token 1, and a creep window that stops short
/// of the head runs the decode rows alone and holds the creep's rows in its
/// residual.
#[derive(Clone)]
struct FourLayers {
    inner: DummyModel,
}

impl ManagedBatchedModel for FourLayers {
    fn maybe_change_dtype(&self, dtype: DType) -> Result<()> {
        self.inner.maybe_change_dtype(dtype)
    }
    fn num_layers(&self) -> usize {
        LAYERS
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
        prefill_inputs: &[Tensor],
        _glue_seqs: &[usize],
        _glue_inputs: &[Tensor],
        _layer_start: usize,
        _layer_end: usize,
        _residual: Option<&mut CreepResidual>,
    ) -> Result<WaveResult> {
        let device = self.inner.device();
        let row = |next: usize| {
            let mut row = vec![0f32; VOCAB];
            row[next] = 50.0;
            Tensor::from_vec(row, (1, VOCAB), device)
        };
        let mut rows = Vec::new();
        for t in decode_inputs {
            for id in t.flatten_all()?.to_vec1::<u32>()? {
                rows.push(row((id as usize + 1) % (VOCAB - 1))?);
            }
        }
        for _ in prefill_inputs {
            rows.push(row(1)?);
        }
        Ok(WaveResult::owned(WaveStep { logits: Some(rows) }))
    }

    fn forward_step(
        &self,
        session: &mut BatchedInferenceSession,
        spec: StepSpec<'_>,
    ) -> Result<StepOutcome> {
        let short = spec.creep.as_ref().is_some_and(|c| c.win_end < LAYERS);
        if !short {
            return stub_step(self, session, spec);
        }
        let StepSpec { decode, creep, .. } = spec;
        let creep = creep.expect("a short window is a creep's");
        if creep.residual.is_none() {
            let rows = creep
                .rows
                .inputs
                .iter()
                .map(|t| t.dims().get(1).copied().unwrap_or(0))
                .sum();
            *creep.residual = Some(stub_residual(rows)?);
        }
        let head = self.forward_wave(
            session,
            decode.seqs,
            decode.inputs,
            &[],
            &[],
            &[],
            &[],
            0,
            LAYERS,
            None,
        );
        Ok(StepOutcome {
            creep_advanced: true,
            creep_us: 0,
            head,
        })
    }

    fn claim_residual(
        &self,
        _session: &BatchedInferenceSession,
        rows: usize,
    ) -> Result<CreepResidual> {
        stub_residual(rows)
    }
}

fn four_layer_scheduler() -> Scheduler {
    let (_tx, rx) = flume::bounded(16);
    Scheduler::new(
        rx,
        Box::new(FourLayers {
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
    )
}

/// A decoding slot of `band` whose last committed token is `last`.
fn decoding(scheduler: &mut Scheduler, band: DecodePriority, last: u32) -> Receiver<TurnEvent> {
    let id = SequenceId(scheduler.session.create_sequence().unwrap());
    let (mut state, events) = boundary_state();
    state.sampling_config = SamplingConfig::argmax();
    state.generated_tokens = TokenBuffer::from(vec![last]);
    scheduler.active_decodes.insert(id, state);
    scheduler
        .sampling_states
        .insert(id, SequenceSamplingState::new(VOCAB, 8));
    scheduler.slot_bands.insert(id, band);
    events
}

/// A prompt of `band` in flight, `tokens` long, and the channel its caller
/// hears it on.
fn prompting(
    scheduler: &mut Scheduler,
    band: DecodePriority,
    tokens: usize,
) -> Receiver<TurnEvent> {
    let id = SequenceId(scheduler.session.create_sequence().unwrap());
    scheduler.slot_bands.insert(id, band);
    scheduler
        .sampling_states
        .insert(id, SequenceSamplingState::new(VOCAB, 8));
    let (event_tx, events) = flume::unbounded();
    let work = PrefillWork {
        tokens: TokenBuffer::from(vec![1u32; tokens]),
        event_tx,
        ..test_prefill_work(id)
    };
    scheduler.active_prefills.push(ActivePrefill {
        work,
        offset: 0,
        next_projection: 0,
        finished: None,
        error: None,
        prefill_start: None,
        stops_every: None,
        awaiting: false,
    });
    events
}

/// What a caller heard of its prefill: each window's layers, and each chunk
/// landing, in order — a repeat collapsed, as the group's completion and the
/// turn's promotion to decode both tell the last landing.
#[derive(Debug, PartialEq, Eq)]
enum Heard {
    Layers(usize, usize),
    Landed(usize, usize),
}

fn heard(events: &Receiver<TurnEvent>) -> Vec<Heard> {
    let mut heard: Vec<Heard> = events
        .try_iter()
        .filter_map(|e| match e {
            TurnEvent::PrefillLayers {
                layers_done,
                layers_total,
            } => Some(Heard::Layers(layers_done, layers_total)),
            TurnEvent::PrefillProgress {
                tokens_done,
                tokens_total,
            } => Some(Heard::Landed(tokens_done, tokens_total)),
            _ => None,
        })
        .collect();
    heard.dedup();
    heard
}

/// One decode step as the decode quantum runs it with nothing owed: the creep
/// and a section chunk free to ride (`run::run_decode_until_budget`).
fn step(scheduler: &mut Scheduler) {
    scheduler.creep_owed = Duration::ZERO;
    scheduler.wave_cohort_advanced = false;
    scheduler.wave_section_advanced = false;
    scheduler.batch_decode_step();
}

/// **Each window a crawling cohort runs is heard by its member.** Beside an
/// `Urgent` decode the `Low` prompt crosses a layer a step — no layer's cost
/// measured yet, a slice is one layer — and it hears the three windows that
/// stop short, then its chunk landing at the fourth.
#[test]
fn a_crawling_member_hears_every_window_then_its_chunk_land() {
    let mut scheduler = four_layer_scheduler();
    let _urgent = decoding(&mut scheduler, DecodePriority::Urgent, 10);
    let reading = prompting(&mut scheduler, DecodePriority::Low, 64);

    for _ in 0..LAYERS {
        step(&mut scheduler);
    }
    assert_eq!(
        heard(&reading),
        vec![
            Heard::Layers(1, 4),
            Heard::Layers(2, 4),
            Heard::Layers(3, 4),
            Heard::Landed(64, 64),
        ]
    );
}

/// **A turn queued behind a crawling cohort is told the work ahead moved**,
/// and its wait reads as movement (`TurnHandle::moves`); with no window run
/// since, the next tell says it did not.
#[test]
fn a_turn_queued_behind_a_crawling_cohort_hears_it_move() {
    let mut scheduler = four_layer_scheduler();
    let _urgent = decoding(&mut scheduler, DecodePriority::Urgent, 10);
    let _reading = prompting(&mut scheduler, DecodePriority::Low, 64);
    let queued_id = SequenceId(scheduler.session.create_sequence().unwrap());
    scheduler.slot_bands.insert(queued_id, DecodePriority::Low);
    let (event_tx, queued) = flume::unbounded();
    scheduler.prefill_queue.push_back(PrefillWork {
        event_tx,
        ..test_prefill_work(queued_id)
    });

    scheduler.report_prefill_waits();
    step(&mut scheduler);
    std::thread::sleep(Duration::from_millis(1_050));
    scheduler.report_prefill_waits();
    let moved: Vec<bool> = queued
        .try_iter()
        .filter_map(|e| match e {
            TurnEvent::PrefillQueued { ahead_moved, .. } => Some(ahead_moved),
            _ => None,
        })
        .collect();
    assert_eq!(moved, vec![true], "a window ran after it queued");

    std::thread::sleep(Duration::from_millis(1_050));
    scheduler.report_prefill_waits();
    let still: Vec<bool> = queued
        .try_iter()
        .filter_map(|e| match e {
            TurnEvent::PrefillQueued { ahead_moved, .. } => Some(ahead_moved),
            _ => None,
        })
        .collect();
    assert_eq!(still, vec![false], "nothing ran since the last tell");
}
