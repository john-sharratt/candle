//! Every exit from a speculative step that keeps none of a block ends what the
//! verify armed (`docs/deltanet_timeline.md` §6.4).
//!
//! A recurrent model's verify rows stay pending past the forward — the block
//! wrote each DeltaNet layer's timeline, not its state — until the rollback
//! folds the rows the step keeps. A step that ends any other way must say so
//! through `abort_verify`, or the rows stay pending into the next step and the
//! model refuses it. A stub drafter records every abort it is sent; each test
//! drives one exit and reads what was aborted.

use super::*;
use crate::guest::Guests;
use candle::{Device, Result};
use candle_nn::kv_cache::ModelGeometry;
use candle_transformers::models::batched_inference::WaveStep;
use candle_transformers::models::delta_net::TIMELINE_ROWS;
use candle_transformers::models::speculative_choice::{SpecRow, TokenChooser};
use candle_transformers::models::verify_wave::VerifyPlan;

const VOCAB: usize = 512;
const DRAFT: usize = 2;

/// Every `abort_verify` call's sequences, in order.
type Aborts = Arc<Mutex<Vec<Vec<usize>>>>;

/// A one-layer stub whose every row predicts `t + 1` and whose drafter
/// proposes that chain, so every proposal is accepted — and whose rollback
/// fails on demand.
#[derive(Clone)]
struct ExitDrafter {
    inner: DummyModel,
    verifying: Arc<Mutex<Vec<usize>>>,
    aborts: Aborts,
    fail_rollback: bool,
    /// Declares recurrent state — the hybrid's shape, whose blocks must fit
    /// the DeltaNet timeline.
    recurrent: bool,
}

impl ExitDrafter {
    fn row_after(&self, t: u32) -> Result<Tensor> {
        let mut row = vec![0f32; VOCAB];
        row[(t + 1) as usize] = 50.0;
        Tensor::from_vec(row, (1, VOCAB), self.inner.device())
    }
}

impl ManagedBatchedModel for ExitDrafter {
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
        prefill_seqs: &[usize],
        prefill_inputs: &[Tensor],
        _glue_seqs: &[usize],
        _glue_inputs: &[Tensor],
        _layer_start: usize,
        _layer_end: usize,
        _residual: Option<&mut CreepResidual>,
    ) -> Result<WaveResult> {
        let mut rows = Vec::new();
        for t in decode_inputs {
            for id in t.flatten_all()?.to_vec1::<u32>()? {
                rows.push(self.row_after(id)?);
            }
        }
        let verifying = self.verifying.lock().unwrap().clone();
        for (seq, t) in prefill_seqs.iter().zip(prefill_inputs) {
            let ids = t.flatten_all()?.to_vec1::<u32>()?;
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
            .map(|&c| (1..=max_len as u32).map(|k| c + k).collect())
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

    fn abort_verify(&self, seqs: &[usize]) {
        self.aborts.lock().unwrap().push(seqs.to_vec());
    }

    fn carries_recurrent_state(&self) -> bool {
        self.recurrent
    }

    fn can_rewind_speculative_block(&self) -> bool {
        true
    }

    fn truncate_sequences(
        &self,
        session: &mut BatchedInferenceSession,
        targets: &[(usize, usize)],
    ) -> Result<()> {
        if self.fail_rollback {
            candle::bail!("the stub's rollback fails on purpose");
        }
        for &(seq, tokens) in targets {
            let at = session.sequence_offset(seq).unwrap_or(0);
            session.advance_sequence(seq, tokens.saturating_sub(at))?;
        }
        Ok(())
    }
}

/// A scheduler over [`ExitDrafter`] with two decoding slots, and the record
/// of what it aborted.
struct Pair {
    scheduler: Scheduler,
    ids: [SequenceId; 2],
    aborts: Aborts,
    _events: [Receiver<TurnEvent>; 2],
}

fn decoding_pair(fail_rollback: bool) -> Pair {
    let aborts: Aborts = Arc::new(Mutex::new(Vec::new()));
    let model = ExitDrafter {
        inner: DummyModel::new(),
        verifying: Arc::new(Mutex::new(Vec::new())),
        aborts: Arc::clone(&aborts),
        fail_rollback,
        recurrent: false,
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
    let mut slot = |last: u32| {
        let id = SequenceId(scheduler.session.create_sequence().unwrap());
        let (mut state, events) = boundary_state();
        state.sampling_config = SamplingConfig::argmax();
        state.generated_tokens = TokenBuffer::from(vec![last]);
        scheduler.active_decodes.insert(id, state);
        scheduler
            .sampling_states
            .insert(id, SequenceSamplingState::new(VOCAB, 8));
        (id, events)
    };
    let (a, ea) = slot(110);
    let (b, eb) = slot(210);
    Pair {
        scheduler,
        ids: [a, b],
        aborts,
        _events: [ea, eb],
    }
}

/// **A rollback that fails keeps none of the blocks**, so every verifying
/// sequence's rows are aborted — and the turns fail.
#[test]
fn a_failed_rollback_aborts_every_verifying_sequence() {
    let Pair {
        mut scheduler,
        ids,
        aborts,
        _events,
    } = decoding_pair(true);
    scheduler.batch_decode_step();

    let aborted = aborts.lock().unwrap().clone();
    let mut last = aborted
        .last()
        .cloned()
        .expect("the failed rollback aborted");
    last.sort_unstable();
    let mut want = vec![ids[0].0, ids[1].0];
    want.sort_unstable();
    assert_eq!(last, want, "every verifying sequence's rows are dropped");
    for id in ids {
        assert!(scheduler.active_decodes[&id].faulted, "the turn failed");
    }
}

/// **A row that commits nothing keeps none of its block**: it is left out of
/// the rollback and its rows alone are aborted — the other sequence's block is
/// rolled back and kept.
#[test]
fn a_row_that_commits_nothing_has_its_block_aborted() {
    let Pair {
        mut scheduler,
        ids,
        aborts,
        _events,
    } = decoding_pair(false);
    // A cut armed before the step stops the sequence at its first position,
    // so it commits nothing.
    scheduler
        .active_decodes
        .get_mut(&ids[1])
        .unwrap()
        .pending_page_cut = true;
    scheduler.batch_decode_step();

    assert_eq!(
        *aborts.lock().unwrap(),
        vec![vec![ids[1].0]],
        "only the row that committed nothing is aborted"
    );
    assert!(scheduler.active_decodes[&ids[1]].faulted);
    assert!(!scheduler.active_decodes[&ids[0]].faulted);
    assert_eq!(
        &scheduler.active_decodes[&ids[0]].generated_tokens[..],
        &[110u32, 111, 112, 113][..],
        "the other sequence keeps its whole block"
    );
}

/// Greedy over each row, read on the host — the stub's rows live there.
struct HostArgmax;

impl TokenChooser for HostArgmax {
    fn choose(&mut self, logits: &Tensor, _rows: &[SpecRow<'_>]) -> Result<Vec<u32>> {
        logits.argmax(1)?.to_vec1::<u32>()
    }
}

/// **The standalone step bounds a recurrent model's block at the timeline**
/// (`docs/deltanet_timeline.md` §10, test 8): eight drafts would make a
/// nine-row block, which no DeltaNet layer can keep a timeline for, so it is
/// refused before anything is drafted — and seven, an eight-row block, runs
/// and keeps every accepted token. A model with no recurrent state is not
/// bounded.
#[test]
fn the_standalone_step_bounds_a_recurrent_block_at_the_timeline() {
    let step = |recurrent: bool, drafts: usize| -> Result<Vec<u32>> {
        let model = ExitDrafter {
            inner: DummyModel::new(),
            verifying: Arc::new(Mutex::new(Vec::new())),
            aborts: Arc::new(Mutex::new(Vec::new())),
            fail_rollback: false,
            recurrent,
        };
        let mut session = make_test_session();
        let seq = session.create_sequence().unwrap();
        let mut kept = Vec::new();
        let mut emits: Vec<Box<dyn FnMut(u32) -> bool + '_>> = vec![Box::new(|t| {
            kept.push(t);
            true
        })];
        model.speculative_decode_step_batch(
            &mut session,
            &[seq],
            &[110],
            &[drafts],
            model.num_layers(),
            &mut HostArgmax,
            &mut emits,
        )?;
        drop(emits);
        Ok(kept)
    };
    let err = step(true, TIMELINE_ROWS).unwrap_err().to_string();
    assert!(err.contains("timeline"), "{err}");
    let kept = step(true, TIMELINE_ROWS - 1).unwrap();
    assert_eq!(kept, (111..=110 + TIMELINE_ROWS as u32).collect::<Vec<_>>());
    assert_eq!(step(false, TIMELINE_ROWS).unwrap().len(), TIMELINE_ROWS + 1);
}
