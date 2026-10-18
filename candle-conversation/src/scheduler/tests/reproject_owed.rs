//! A reprojected turn whose new view the span cannot fork waits for it.
//!
//! The rebuilt view's fork of its parent's recurrent state is a reprojection's
//! last claim, made after the gap-fill forward. Refused there for ground, it
//! used to end the turn: on the RTX 3090 (2026-10-18), "no region for recurrent
//! state after 20 of 36 slots" with the weight side at its floor. The turn now
//! parks beside its parent and is carved on a later pass of the loop.

use super::*;
use crate::guest::Guests;
use crate::scheduler::reproject_owed::OwedView;
use candle::{Device, Result};
use candle_nn::kv_cache::{ModelGeometry, SPAN_FULL_MARKER};
use candle_transformers::models::batched_inference::WaveStep;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// A one-layer stub whose fork of a parent's recurrent state is refused for
/// ground while `refusals` is above zero.
#[derive(Clone)]
struct RefusingFork {
    inner: DummyModel,
    refusals: Arc<AtomicUsize>,
}

impl ManagedBatchedModel for RefusingFork {
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
        _decode_inputs: &[Tensor],
        _prefill_seqs: &[usize],
        _prefill_inputs: &[Tensor],
        _glue_seqs: &[usize],
        _glue_inputs: &[Tensor],
        _layer_start: usize,
        _layer_end: usize,
        _residual: Option<&mut CreepResidual>,
    ) -> Result<WaveResult> {
        Ok(WaveResult::owned(WaveStep { logits: None }))
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

    fn fork_recurrent(&self, _parent: usize, _child: usize) -> Result<()> {
        let left = self.refusals.load(Ordering::Relaxed);
        if left > 0 {
            self.refusals.store(left - 1, Ordering::Relaxed);
            candle::bail!(
                "slot arena: no region for recurrent state after 20 of 36 slots of 3859968 B \
                 — {SPAN_FULL_MARKER} — the weight side is at its own floor"
            );
        }
        Ok(())
    }
}

/// A scheduler over a parent slot whose views' forks are refused `refusals`
/// times, and the turn a reprojection of it owes a view: its old view already
/// freed, its state out of every map the decode loop reads.
fn owed_turn(
    refusals: usize,
    since: Instant,
) -> (Scheduler, SequenceId, OwedView, Receiver<TurnEvent>) {
    let model = RefusingFork {
        inner: DummyModel::new(),
        refusals: Arc::new(AtomicUsize::new(refusals)),
    };
    let (_tx, rx) = flume::bounded(16);
    let mut scheduler = Scheduler::new(
        rx,
        Box::new(model),
        make_test_session(),
        make_dummy_tokenizer(),
        vec![0].into(),
        512,
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
    let parent = SequenceId(scheduler.session.create_sequence().unwrap());
    let (decode_state, events) = boundary_state();
    let owed = OwedView {
        view_id: SequenceId(97),
        parent_id: parent,
        decode_state,
        sampling_state: None,
        turn_start_parent_blocks: 0,
        question_tokens: 0,
        applied_identity: None,
        since,
    };
    (scheduler, parent, owed, events)
}

fn errors(events: &Receiver<TurnEvent>) -> Vec<String> {
    events
        .try_iter()
        .filter_map(|e| match e {
            TurnEvent::Error(e) => Some(e.to_string()),
            _ => None,
        })
        .collect()
}

/// **A fork refused for ground parks the turn, and a later pass carves its
/// view** — the turn re-bound to it whole, having heard nothing.
#[test]
fn a_view_refused_its_fork_for_ground_waits_and_is_carved_later() {
    let (mut scheduler, parent, owed, events) = owed_turn(2, Instant::now());
    let owed = match scheduler.carve_owed_view(owed) {
        Ok(Err(owed)) => owed,
        Ok(Ok(_)) => panic!("the refused fork carved a view"),
        Err(e) => panic!("a refusal for ground abandoned the turn: {e}"),
    };
    scheduler.owed_views.push(owed);
    assert!(scheduler.active_decodes.is_empty());
    assert!(
        scheduler.owes_view(parent),
        "its parent reads as a turn in flight"
    );

    scheduler.retry_owed_views();
    assert_eq!(scheduler.owed_views.len(), 1, "still refused: still owed");

    scheduler.retry_owed_views();
    assert!(
        scheduler.owed_views.is_empty(),
        "carved once ground came back"
    );
    let (&view, _) = scheduler
        .active_decodes
        .iter()
        .next()
        .expect("the turn decodes again");
    let state = &scheduler.turn_views[&view];
    assert_eq!(state.parent_id, parent);
    assert!(state.state_forked);
    assert!(errors(&events).is_empty(), "the turn heard no error");
}

/// **A view owed past its patience fails the turn with the refusal**, rather
/// than holding a caller on ground that never comes back.
#[test]
fn a_view_owed_past_its_patience_fails_the_turn() {
    let long_ago = Instant::now() - Duration::from_secs(61);
    let (mut scheduler, parent, owed, events) = owed_turn(1, long_ago);
    assert!(scheduler.carve_owed_view(owed).is_err());
    assert!(scheduler.active_decodes.is_empty());
    assert!(!scheduler.owes_view(parent));
    let errs = errors(&events);
    assert_eq!(errs.len(), 1);
    assert!(
        errs[0].contains(SPAN_FULL_MARKER),
        "the turn hears what refused it: {}",
        errs[0]
    );
}

/// **A turn whose caller stopped listening is not carved**: its owed view
/// goes, and no slot is taken for it.
#[test]
fn a_view_owed_to_a_caller_that_left_is_dropped() {
    let (mut scheduler, parent, owed, events) = owed_turn(0, Instant::now());
    scheduler.owed_views.push(owed);
    drop(events);
    scheduler.retry_owed_views();
    assert!(scheduler.owed_views.is_empty());
    assert!(scheduler.active_decodes.is_empty());
    assert!(!scheduler.owes_view(parent));
}
