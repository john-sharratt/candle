//! Speculative decode inside a grammar's free-text span — a think block.
//!
//! A stub model predicts a fixed successor for every token and drafts exactly
//! that chain, so every proposal is accepted unless the scheduler stops the
//! block. What each test checks is therefore the scheduler's own decision: that
//! a sequence in a think span drafts and commits the whole accepted block, and
//! that the block ends at the token that closes the span, with the KV holding
//! exactly the committed tokens.

use super::*;
use crate::guest::Guests;
use crate::stencil::{
    compile, compile_think_tree, TestVocab, ThinkMode, ThinkSteerEnvelope, Vocab,
};
use candle::{Device, Result};
use candle_nn::kv_cache::{ModelGeometry, TIER_REFUSED_MARKER};
use candle_transformers::models::batched_inference::WaveStep;
use candle_transformers::models::verify_wave::VerifyPlan;
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

thread_local! {
    /// Waves the stub refuses for ground before it runs one, on this test's
    /// thread.
    static REFUSALS: Cell<usize> = const { Cell::new(0) };
}

/// The blocks a [`ChainDrafter`] was asked to verify, in order.
type Verified = Arc<Mutex<Vec<Vec<u32>>>>;

const VOCAB: usize = 512;
const THINK_OPEN: u32 = 300;
const THINK_CLOSE: u32 = 301;
/// The token after which the stub writes `</think>`.
const CLOSES_AFTER: u32 = 101;
const DRAFT: usize = 2;

/// The token the stub predicts after `t`.
fn successor(t: u32) -> u32 {
    if t == CLOSES_AFTER {
        THINK_CLOSE
    } else {
        t + 1
    }
}

/// A one-layer stub whose every row predicts [`successor`] and whose drafter
/// proposes the same chain, recording each block it is asked to verify.
#[derive(Clone)]
struct ChainDrafter {
    inner: DummyModel,
    verified: Verified,
    /// The sequences the current step verifies — scored at every position. Any
    /// other prefill member is a plain span and gets one row, after its last
    /// token, as a real model's prefill does.
    verifying: Arc<Mutex<Vec<usize>>>,
    /// Whole-model forwards run, so a test can count what a step cost.
    forwards: Arc<AtomicUsize>,
}

impl ChainDrafter {
    fn row_after(&self, t: u32) -> Result<Tensor> {
        let mut row = vec![0f32; VOCAB];
        row[successor(t) as usize] = 50.0;
        Tensor::from_vec(row, (1, VOCAB), self.inner.device())
    }

    fn ids(t: &Tensor) -> Result<Vec<u32>> {
        t.flatten_all()?.to_vec1::<u32>()
    }
}

impl ManagedBatchedModel for ChainDrafter {
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

    /// One row per decode member, one per token of a verifying member, and one
    /// after the last token of a plain prefill span.
    #[allow(clippy::too_many_arguments)]
    fn forward_wave(
        &self,
        _session: &mut BatchedInferenceSession,
        _decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        prefill_seqs: &[usize],
        prefill_inputs: &[Tensor],
        _glue_seqs: &[usize],
        _glue_inputs: &[Tensor],
        _layer_start: usize,
        _layer_end: usize,
        _residual: Option<&mut CreepResidual>,
    ) -> Result<WaveResult> {
        if REFUSALS.with(|r| r.replace(r.get().saturating_sub(1))) > 0 {
            candle::bail!("{TIER_REFUSED_MARKER}: the stub has no ground for this wave");
        }
        self.forwards.fetch_add(1, Ordering::Relaxed);
        let mut rows = Vec::new();
        for t in decode_inputs {
            for id in Self::ids(t)? {
                rows.push(self.row_after(id)?);
            }
        }
        let verifying = self.verifying.lock().unwrap().clone();
        for (seq, t) in prefill_seqs.iter().zip(prefill_inputs) {
            let ids = Self::ids(t)?;
            if verifying.contains(seq) {
                for id in ids {
                    rows.push(self.row_after(id)?);
                }
            } else if let Some(&last) = ids.last() {
                rows.push(self.row_after(last)?);
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

    fn draft_budget(&self, _width: usize) -> usize {
        DRAFT
    }

    fn speculative_draft(
        &self,
        _session: &mut BatchedInferenceSession,
        _seqs: &[usize],
        committed: &[u32],
        max_len: usize,
    ) -> Result<Vec<Vec<u32>>> {
        Ok(committed
            .iter()
            .map(|&c| {
                let mut chain = Vec::with_capacity(max_len);
                let mut t = c;
                for _ in 0..max_len {
                    t = successor(t);
                    chain.push(t);
                }
                chain
            })
            .collect())
    }

    fn begin_verify(
        &self,
        _session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        seqs: &[usize],
        blocks: &[Vec<u32>],
        _budget: usize,
    ) -> Result<Option<VerifyPlan>> {
        self.verified.lock().unwrap().extend(blocks.iter().cloned());
        *self.verifying.lock().unwrap() = seqs.to_vec();
        let device = self.inner.device();
        Ok(Some(VerifyPlan {
            decode_seqs: plain.iter().map(|&(s, _)| s).collect(),
            decode_inputs: plain
                .iter()
                .map(|&(_, t)| Tensor::from_vec(vec![t], (1, 1), device))
                .collect::<Result<_>>()?,
            verify_seqs: seqs.to_vec(),
            verify_inputs: blocks
                .iter()
                .map(|b| Tensor::from_vec(b.clone(), (1, b.len()), device))
                .collect::<Result<_>>()?,
            rows: plain.len() + blocks.iter().map(Vec::len).sum::<usize>(),
        }))
    }

    /// Splits the rows back out. The sequences are advanced by the rollback
    /// below rather than here, so the session's offset only ever counts what
    /// the scheduler kept.
    fn end_verify(
        &self,
        _session: &mut BatchedInferenceSession,
        plain: &[(usize, u32)],
        _seqs: &[usize],
        blocks: &[Vec<u32>],
        logits: Vec<Tensor>,
    ) -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
        self.verifying.lock().unwrap().clear();
        let (plain_rows, mut rest) = logits.split_at(plain.len());
        let mut per_block = Vec::with_capacity(blocks.len());
        for b in blocks {
            let (block_rows, tail) = rest.split_at(b.len());
            per_block.push(block_rows.to_vec());
            rest = tail;
        }
        Ok((plain_rows.to_vec(), per_block))
    }

    /// Moves each sequence to the length the scheduler kept.
    fn truncate_sequences(
        &self,
        session: &mut BatchedInferenceSession,
        targets: &[(usize, usize)],
    ) -> Result<()> {
        for &(seq, tokens) in targets {
            let at = session.sequence_offset(seq).unwrap_or(0);
            if tokens < at {
                candle::bail!("rollback to {tokens} below the {at} this stub has written");
            }
            session.advance_sequence(seq, tokens - at)?;
        }
        Ok(())
    }
}

fn think_tree() -> Arc<StencilTree> {
    let vocab = TestVocab::new()
        .with_special("<think>", THINK_OPEN)
        .with_special("</think>", THINK_CLOSE);
    let env = ThinkSteerEnvelope {
        think_open: THINK_OPEN,
        think_close: THINK_CLOSE,
        eos: vocab.eos(),
        after_close: "",
    };
    Arc::new(compile(&compile_think_tree(ThinkMode::Balanced, &env), &vocab).unwrap())
}

/// One decoding slot under test, and what reads it.
struct Slot {
    scheduler: Scheduler,
    id: SequenceId,
    verified: Verified,
    forwards: Arc<AtomicUsize>,
    /// The turn's event stream. Held so the slot's token sends succeed — a
    /// closed stream finishes the turn at its first token.
    _events: Receiver<TurnEvent>,
}

/// A scheduler over [`ChainDrafter`] with one decoding slot whose last
/// committed token is `last`. With `in_think_span`, the slot is inside a think
/// block's free span, exactly as `play_stencil_runs` leaves a sequence
/// there before a decode step.
fn decoding_slot(last: u32, in_think_span: bool) -> Slot {
    let verified = Arc::new(Mutex::new(Vec::new()));
    let forwards = Arc::new(AtomicUsize::new(0));
    let model = ChainDrafter {
        inner: DummyModel::new(),
        verified: Arc::clone(&verified),
        verifying: Arc::new(Mutex::new(Vec::new())),
        forwards: Arc::clone(&forwards),
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
    let (id, events) = add_slot(&mut scheduler, last, in_think_span);
    Slot {
        scheduler,
        id,
        verified,
        forwards,
        _events: events,
    }
}

/// Add a decoding slot to `scheduler` whose last committed token is `last`,
/// inside a think block's free span when `in_think_span`. Returns its id and
/// its event stream, which the caller holds for as long as the slot decodes.
fn add_slot(
    scheduler: &mut Scheduler,
    last: u32,
    in_think_span: bool,
) -> (SequenceId, Receiver<TurnEvent>) {
    let id = SequenceId(scheduler.session.create_sequence().unwrap());
    let (mut state, events) = boundary_state();
    state.sampling_config = SamplingConfig::argmax();
    state.generated_tokens = TokenBuffer::from(vec![last]);
    if in_think_span {
        let mut driver = StencilDriver::new(think_tree());
        let action = loop {
            match driver.step() {
                StepMask::Prefill(_) => continue,
                action => break action,
            }
        };
        assert!(matches!(action, StepMask::Free { .. }), "the block's span");
        state.stencil = Some(driver);
        state.pending_mask = Some(action);
    }
    scheduler.active_decodes.insert(id, state);
    scheduler
        .sampling_states
        .insert(id, SequenceSamplingState::new(VOCAB, 8));
    (id, events)
}

/// **The speedup is real inside a think block**: the sequence drafts, the
/// whole block is verified, and every accepted token is committed — three
/// tokens from one forward, where a plain step commits one.
#[test]
fn a_think_span_commits_every_drafted_token_it_accepts() {
    let Slot {
        mut scheduler,
        id: slot,
        verified,
        _events,
        ..
    } = decoding_slot(110, true);
    scheduler.batch_decode_step();

    assert_eq!(*verified.lock().unwrap(), vec![vec![110u32, 111, 112]]);
    let state = &scheduler.active_decodes[&slot];
    assert_eq!(&state.generated_tokens[..], &[110u32, 111, 112, 113][..]);
    assert_eq!(
        scheduler.session.sequence_offset(slot.0),
        Some(3),
        "the input and both drafted tokens are written; 113 rides the next step"
    );
    assert_eq!(
        state.pending_forward(),
        &[113u32][..],
        "only the last committed token is waiting to be forwarded"
    );
    let driver = state.stencil.as_ref().expect("the block is still open");
    assert!(driver.mid_free_span());
}

/// Free decode drafts exactly as it did before a grammar could.
#[test]
fn free_decode_commits_every_drafted_token_it_accepts() {
    let Slot {
        mut scheduler,
        id: slot,
        verified,
        _events,
        ..
    } = decoding_slot(110, false);
    scheduler.batch_decode_step();

    assert_eq!(*verified.lock().unwrap(), vec![vec![110u32, 111, 112]]);
    let state = &scheduler.active_decodes[&slot];
    assert_eq!(&state.generated_tokens[..], &[110u32, 111, 112, 113][..]);
    assert_eq!(scheduler.session.sequence_offset(slot.0), Some(3));
}

/// **The block ends at the close the span drops.** The model's `</think>` is
/// sampled and accepted as a proposal, but the span suppresses it: it is not
/// committed, the KV keeps only what was, and the tree then writes its own
/// closing tag — once, with nothing forwarded twice.
#[test]
fn a_think_span_block_ends_at_the_close_it_drops() {
    let Slot {
        mut scheduler,
        id: slot,
        verified,
        forwards,
        _events,
    } = decoding_slot(100, true);
    scheduler.batch_decode_step();

    assert_eq!(
        *verified.lock().unwrap(),
        vec![vec![100u32, 101, THINK_CLOSE]]
    );
    let state = &scheduler.active_decodes[&slot];
    assert_eq!(
        &state.generated_tokens[..],
        &[100u32, 101][..],
        "the dropped close is not committed"
    );
    assert_eq!(
        scheduler.session.sequence_offset(slot.0),
        Some(2),
        "the rollback keeps 100 and 101 and nothing past them"
    );
    assert!(
        state.pending_forward().is_empty(),
        "101 is in the KV, so nothing is waiting to be forwarded"
    );
    assert!(state.pending_mask.is_none());
    assert!(!state.stencil.as_ref().unwrap().mid_free_span());

    // **The tree's closing tag is a static run, and it costs no forward of its
    // own.** Playing it records the row — nothing already written goes with it
    // — and forwards nothing.
    let before = forwards.load(Ordering::Relaxed);
    scheduler.play_stencil_runs();
    assert_eq!(
        forwards.load(Ordering::Relaxed) - before,
        0,
        "playing a run forwards nothing"
    );
    assert_eq!(
        &scheduler.active_decodes[&slot].generated_tokens[..],
        &[100u32, 101, THINK_CLOSE][..]
    );
    assert_eq!(scheduler.session.sequence_offset(slot.0), Some(2));

    // The decode step's wave writes it, and the token after it is sampled from
    // that wave's row for it, rather than the tag riding a second forward that
    // reads the same distribution.
    scheduler.batch_decode_step();
    assert_eq!(
        forwards.load(Ordering::Relaxed) - before,
        1,
        "the static run and the token after it share the step's one forward"
    );
    let after_close = successor(THINK_CLOSE);
    let state = &scheduler.active_decodes[&slot];
    assert_eq!(
        &state.generated_tokens[..],
        &[100u32, 101, THINK_CLOSE, after_close][..],
        "the run, then the token sampled from the run's own row"
    );
    assert_eq!(
        scheduler.session.sequence_offset(slot.0),
        Some(3),
        "the closing tag is written; the token after it is not yet"
    );
    assert_eq!(
        state.pending_forward(),
        &[after_close][..],
        "only the sampled token waits to be forwarded"
    );
    assert!(!scheduler.holds_stencil_row(slot));
}

/// **A static run rides the wave a decode row is in.** One sequence's closing
/// tag and another's drafted block go out in one forward: the run is scored at
/// its last row, the block at every row, and each sequence commits from its own.
#[test]
fn a_static_run_shares_the_wave_of_another_sequences_decode() {
    let Slot {
        mut scheduler,
        id: closing,
        forwards,
        _events,
        ..
    } = decoding_slot(100, true);
    // To the close the span drops, which leaves the tag to play.
    scheduler.batch_decode_step();
    let (decoding, _decoding_events) = add_slot(&mut scheduler, 200, false);

    scheduler.play_stencil_runs();
    let before = forwards.load(Ordering::Relaxed);
    scheduler.batch_decode_step();
    assert_eq!(
        forwards.load(Ordering::Relaxed) - before,
        1,
        "the run and the other sequence's block share one forward"
    );
    assert_eq!(
        &scheduler.active_decodes[&closing].generated_tokens[..],
        &[100u32, 101, THINK_CLOSE, successor(THINK_CLOSE)][..]
    );
    assert_eq!(scheduler.session.sequence_offset(closing.0), Some(3));
    assert_eq!(
        &scheduler.active_decodes[&decoding].generated_tokens[..],
        &[200u32, 201, 202, 203][..],
        "the block's rows are its own, not shifted by the run's"
    );
    assert_eq!(scheduler.session.sequence_offset(decoding.0), Some(3));
}

/// **A wave refused for ground keeps the run for the next.** The run's tokens
/// are already recorded, so failing the turn would throw a reply away over
/// ground the next step finds, and dropping the row would seal tokens with no
/// K/V behind them. The retry writes it exactly once.
#[test]
fn a_refused_wave_keeps_the_static_run_for_the_next_step() {
    let Slot {
        mut scheduler,
        id: slot,
        forwards,
        _events,
        ..
    } = decoding_slot(100, true);
    scheduler.batch_decode_step();
    scheduler.play_stencil_runs();

    REFUSALS.with(|r| r.set(1));
    scheduler.batch_decode_step();
    let state = &scheduler.active_decodes[&slot];
    assert!(!state.finished, "a refusal is not the turn's failure");
    assert!(
        scheduler.holds_stencil_row(slot),
        "the row waits for the retry"
    );
    assert_eq!(&state.generated_tokens[..], &[100u32, 101, THINK_CLOSE][..]);
    assert_eq!(scheduler.session.sequence_offset(slot.0), Some(2));

    let before = forwards.load(Ordering::Relaxed);
    scheduler.play_stencil_runs();
    scheduler.batch_decode_step();
    assert_eq!(forwards.load(Ordering::Relaxed) - before, 1);
    assert!(!scheduler.holds_stencil_row(slot));
    assert_eq!(
        &scheduler.active_decodes[&slot].generated_tokens[..],
        &[100u32, 101, THINK_CLOSE, successor(THINK_CLOSE)][..],
        "the run is written once and the token after it sampled once"
    );
    assert_eq!(scheduler.session.sequence_offset(slot.0), Some(3));
}
