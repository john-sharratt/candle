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
use candle_nn::kv_cache::ModelGeometry;
use candle_transformers::models::batched_inference::WaveStep;
use candle_transformers::models::verify_wave::VerifyPlan;

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

    /// One row per decode member and one per token of every prefill-slot
    /// member — a verify block is scored at every position.
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
        _residual_in: Option<Tensor>,
    ) -> Result<WaveResult> {
        let mut rows = Vec::new();
        for t in decode_inputs {
            for id in Self::ids(t)? {
                rows.push(self.row_after(id)?);
            }
        }
        for t in prefill_inputs {
            for id in Self::ids(t)? {
                rows.push(self.row_after(id)?);
            }
        }
        Ok(WaveResult::owned(WaveStep {
            residual: None,
            logits: Some(rows),
        }))
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
    /// The turn's event stream. Held so the slot's token sends succeed — a
    /// closed stream finishes the turn at its first token.
    _events: Receiver<TurnEvent>,
}

/// A scheduler over [`ChainDrafter`] with one decoding slot whose last
/// committed token is `last`. With `in_think_span`, the slot is inside a think
/// block's free span, exactly as `inject_stencil_prefills` leaves a sequence
/// there before a decode step.
fn decoding_slot(last: u32, in_think_span: bool) -> Slot {
    let verified = Arc::new(Mutex::new(Vec::new()));
    let model = ChainDrafter {
        inner: DummyModel::new(),
        verified: Arc::clone(&verified),
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
    Slot {
        scheduler,
        id,
        verified,
        _events: events,
    }
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

    // The tree's closing tag is a one-token static: it rides the next decode,
    // with no prefill of anything already written.
    scheduler.inject_stencil_prefills();
    let state = &scheduler.active_decodes[&slot];
    assert_eq!(&state.generated_tokens[..], &[100u32, 101, THINK_CLOSE][..]);
    assert_eq!(scheduler.session.sequence_offset(slot.0), Some(2));
}
