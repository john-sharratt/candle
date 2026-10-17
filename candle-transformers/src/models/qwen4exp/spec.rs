//! Speculative verify for `qwen4exp`: rewinding three recurrences that have no
//! suffix to remove.
//!
//! A speculative step runs a block of proposed tokens through one forward and
//! then learns how many the model agrees with. For the paged K/V that is free —
//! it is append-only, so truncating to the accepted length erases exactly the
//! rejected tokens. This stack carries three things that truncation cannot
//! reach, and every one of them is silent when it is wrong:
//!
//! | Carried state | Why it cannot be truncated | How it rewinds |
//! |---|---|---|
//! | GDN `S` (DeltaNet) | a running sum over every token, no per-token decomposition | never written by the block: fold the accepted rows of its timeline |
//! | PLE conv history | a sliding window of derived rows, not of tokens | slice the window that the accepted rows would have left |
//! | QSA index cache | pooled block keys, appended as blocks complete | restore the entering snapshot, re-append the accepted rows' keys |
//!
//! # The GDN half writes nothing it would have to undo
//!
//! A verify block is a short span, so each DeltaNet layer reads its state and
//! writes the block's **timeline** — the per-row generators of its advance —
//! and stays pending past the forward's commit (`docs/deltanet_timeline.md`).
//! Once the accept walk knows the block keeps `m` rows, the fold applies the
//! first `m`; nothing is replayed, and a sequence that keeps none discards them.
//! It is the hybrid's fold (`fold_stores`), one launch over the cohort.
//!
//! # The other two keep their snapshots
//!
//! The PLE history and the index caches are snapshotted at wave entry for the
//! failure bracket that rolls a failed wave back ([`super::wave`]). A verify
//! wave keeps those snapshots instead of dropping them, so the "state the block
//! was entered with" costs nothing extra; the rewind is bookkeeping over rows
//! the wave already computed: the PLE window is a slice, and the index cache
//! re-appends through [`IndexCache::append`] — the same call the wave made.
//! Their rows sit in [`CaptureRows`], one row per verify row of the cohort.
//!
//! # Cost
//!
//! For a block of `k` proposals accepted at `m`: one fold launch over the
//! cohort's DeltaNet layers, one narrow, and one `append` per attention layer,
//! against a whole forward's 48 layers of projections, attention and a
//! 512-expert MoE.

use std::collections::HashMap;

#[cfg(feature = "cuda")]
use candle::Device;
use candle::{Result, Tensor};

use super::capture_rows::{CaptureRows, CaptureWidths};
use super::engine::GpuLayerMix;
use super::indexer::{append_wave, AppendSpan, IndexCache, IndexSnapshot};
use super::ple::PleState;
use super::qsa::IndexerWeights;
use super::wave::Qwen4ExpBatched;
use crate::models::batched_inference::BatchedInferenceSession;
use crate::models::delta_net::{fold_stores, LayerKind, RecurrentStateStore};

/// One verifying sequence's rewind material.
#[derive(Default)]
pub struct SeqStash {
    /// Where this sequence's rows sit in the verify wave, and how many.
    pub row: usize,
    pub len: usize,
    /// The block's tokens, for the PLE hash window — which is a function of
    /// token ids alone, so it rewinds on the host.
    pub tokens: Vec<u32>,
    /// State entering the block.
    pub ple_entering: Option<PleState>,
    pub qsa_entering: Vec<IndexSnapshot>,
    /// The block's own rows: what the recurrences consumed.
    pub ple_rows: Option<Tensor>,
    /// Per KV layer, `[len, indexer_head_dim]` — the raw projected index keys.
    pub qsa_keys: Vec<Option<Tensor>>,
    /// `[len, hc_dim]` — the trunk residual the lockstep head pass ran over,
    /// which is what the head's seed is drawn from.
    ///
    /// Stashed only for a verifying sequence, because only a verifying sequence
    /// can accept a prefix and need a row other than the last. See
    /// `head_wave_pass`.
    pub head_rows: Option<Tensor>,
}

/// Everything a verify wave has to capture, armed for exactly the sequences
/// whose blocks are being verified.
///
/// `None` on the model means no verify is in flight and every capture site is a
/// single `is_none` check — which is what keeps a plain decode wave paying
/// nothing for machinery only a speculative one uses.
pub struct SpecCapture {
    /// Per verifying sequence — the armed cohort, whose DeltaNet layers stay
    /// pending until the rewind.
    pub seqs: HashMap<usize, SeqStash>,
    /// The cohort's kept rows: one row per verify row, each sequence's block at
    /// its `row`, packed in wave order.
    pub rows: CaptureRows,
}

impl SpecCapture {
    /// Arm for `blocks` — `(sequence, block length)`, in wave order — over
    /// row buffers already sized for the cohort
    /// ([`Qwen4ExpBatched::capture_rows_for`]).
    pub fn new(blocks: &[(usize, usize)], rows: CaptureRows) -> Result<Self> {
        let total: usize = blocks.iter().map(|&(_, len)| len).sum();
        let cap = rows.capacity()?;
        if total > cap {
            candle::bail!(
                "qwen4exp verify: a {total}-row cohort against {cap}-row capture buffers"
            );
        }
        let mut seqs = HashMap::with_capacity(blocks.len());
        let mut row = 0usize;
        for &(seq, len) in blocks {
            if seqs
                .insert(
                    seq,
                    SeqStash {
                        row,
                        len,
                        ..SeqStash::default()
                    },
                )
                .is_some()
            {
                candle::bail!("qwen4exp verify: sequence {seq} carries two blocks in one cohort");
            }
            row += len;
        }
        Ok(Self { seqs, rows })
    }

    /// Record the entering state, taken from the snapshots the wave's failure
    /// bracket already holds — so a verify pays no extra snapshot.
    /// **The first segment's snapshot wins.** A verify block is not always one
    /// `forward_wave` call: co-batched with a creep group it is run as one call
    /// per layer window, and the scheduler drives exactly that — a verify block
    /// rides the prefill slot in every segment. Each of those calls snapshots
    /// and lands here, but only the first snapshot is the state the block was
    /// *entered* with; every later one is taken after the windows below it have
    /// already advanced the recurrences.
    ///
    /// Assigning unconditionally recorded the last, so a partial accept rewound
    /// from a post-advance base: `rewind_row_state` would build
    /// `cat([entering.conv_hist, ple_rows])` out of a `conv_hist` that already
    /// contained the block's rows — selecting a window shifted by `len` — and
    /// for every attention layer below the cursor it would `restore` a cache
    /// still holding the block's keys and then append `kept` more, duplicating
    /// them permanently so later QSA selections score blocks that were never
    /// committed.
    pub fn take_entering(&mut self, seq: usize, ple: PleState, qsa: Vec<IndexSnapshot>) {
        if let Some(s) = self.seqs.get_mut(&seq) {
            if s.ple_entering.is_none() {
                s.ple_entering = Some(ple);
                s.qsa_entering = qsa;
            }
        }
    }
}

impl Qwen4ExpBatched {
    /// The capture's row buffers for a cohort of `rows` verify rows: the ones
    /// the last step used when they are wide enough, new ones sized for `rows`
    /// otherwise.
    ///
    /// **Before the forward opens**, never inside it: a wave's storage is
    /// claimed up front. The buffers are reused across steps and replaced only
    /// when a wider cohort arrives — the old ones released first, so two
    /// cohorts' buffers are never held at once.
    pub(super) fn capture_rows_for(&self, rows: usize) -> Result<CaptureRows> {
        let mut slot = self
            .parked_rows
            .lock()
            .map_err(|_| candle::Error::Msg("parked capture rows lock poisoned".into()))?;
        let fits = match slot.as_ref() {
            Some(r) => r.capacity()? >= rows,
            None => false,
        };
        if fits {
            return Ok(slot.take().expect("checked above"));
        }
        drop(slot.take());
        // At least one row: buffers of none would be zero-width tensors — a
        // round trip per buffer for memory that holds nothing.
        let rows = rows.max(1);
        let cfg = &self.model.cfg;
        let widths = CaptureWidths {
            hc: cfg.hc.count,
            n_embd: cfg.hidden_size,
            kv_layers: self.kv_layer_count(),
            index_dim: cfg.indexer.head_dim,
        };
        CaptureRows::new(rows, widths, &self.model.device)
    }

    /// Hand a consumed capture's row buffers back for the next step: whatever
    /// they held was good for the one rewind the capture was armed for.
    pub(super) fn park_capture(&self, cap: SpecCapture) {
        if let Ok(mut slot) = self.parked_rows.lock() {
            *slot = Some(cap.rows);
        }
    }

    /// Release a capture whose step keeps none of its blocks: every armed
    /// sequence's pending timelines are dropped — its GDN state stands as the
    /// step entered it — and the row buffers are parked.
    pub(super) fn release_capture(&self, cap: SpecCapture) -> Result<()> {
        let discarded = self
            .recurrent
            .write()
            .map_err(|_| candle::Error::Msg("recurrent lock poisoned".into()))
            .map(|mut rec| {
                for seq in cap.seqs.keys() {
                    if let Some(store) = rec.get_mut(seq) {
                        store.discard_pending();
                    }
                }
            });
        self.park_capture(cap);
        discarded
    }

    /// The KV layers a sequence holds an index cache for: every trunk
    /// attention layer, and the draft head's after them.
    fn kv_layer_count(&self) -> usize {
        let cfg = &self.model.cfg;
        let trunk = cfg
            .layer_kinds
            .iter()
            .filter(|k| !matches!(k, LayerKind::DeltaNet))
            .count();
        trunk + usize::from(self.model.mtp.is_some())
    }

    /// Roll every target back to exactly `tokens` tokens — K/V and all three
    /// recurrences — releasing the capture when the cohort is done.
    ///
    /// Called only by the speculative verify path. A target already at its
    /// length is the common case, not a corner one: every step ends by
    /// reconciling each sequence to what it kept, and a step that kept
    /// everything it wrote reconciles to the offset it already stands at.
    pub(super) fn rewind_cohort(
        &self,
        session: &mut BatchedInferenceSession,
        targets: &[(usize, usize)],
    ) -> Result<()> {
        let mut guard = self
            .verify
            .write()
            .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?;

        // **Taken, not borrowed.** A capture is good for exactly one step, and
        // a failed rewind's must not survive to rewind a later one. The borrow
        // left the capture armed on every `?` below, so a cohort that bailed
        // partway — some sequences rewound, none truncated — kept a capture
        // that still read as a valid rewind point for the next
        // `truncate_sequences`. Consuming it up front makes a later rewind
        // refuse outright, which is the honest report of a step that failed.
        // qwen35 consumes its armed cohort the same way (`qwen35/batched.rs`).
        //
        // Without a capture there is no block in flight, so the only honest
        // rewind is none at all — and a non-trivial target here would be a
        // recurrence silently left where the rejected tokens put it.
        let Some(cap) = guard.take() else {
            for &(seq, tokens) in targets {
                if session.sequence_offset(seq) != Some(tokens) {
                    candle::bail!(
                        "qwen4exp rewind: sequence {seq} asked to move to {tokens} tokens with \
                         no verify capture in flight — the recurrent state has no rewind point"
                    );
                }
            }
            return Ok(());
        };

        // How many of each block's rows survived. The wave advanced the
        // sequence by the whole block, so its start is `offset - len`.
        let mut jobs: Vec<(usize, usize)> = Vec::with_capacity(targets.len());
        let planned = (|| -> Result<()> {
            for &(seq, tokens) in targets {
                let Some(stash) = cap.seqs.get(&seq) else {
                    // A plain decode row: it wrote one token and kept it.
                    if session.sequence_offset(seq) != Some(tokens) {
                        candle::bail!(
                            "qwen4exp rewind: sequence {seq} did not verify a block this step \
                             but is asked to move to {tokens} tokens"
                        );
                    }
                    continue;
                };
                let offset = session.sequence_offset(seq).unwrap_or(0);
                let base = offset.checked_sub(stash.len).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "qwen4exp rewind: sequence {seq} stands at {offset} tokens after a \
                         {}-row block, which cannot have started before zero",
                        stash.len
                    ))
                })?;
                let kept = tokens.checked_sub(base).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "qwen4exp rewind: sequence {seq} asked to move to {tokens} tokens, \
                         before the block this capture covers (starts at {base}) — a fold \
                         covers the block whose timeline it holds and nothing else"
                    ))
                })?;
                jobs.push((seq, kept));
            }
            Ok(())
        })();
        if let Err(e) = planned {
            self.release_capture(cap)?;
            return Err(e);
        }

        // **Recorded as one wave, like the draft walk.** Everything below is a
        // chain of small launches — the DeltaNet fold; per attention layer an
        // index append; the seed rewrites — with no host decision between them.
        // Issued eagerly they left the GPU idle between every pair: measured on
        // Flash-Next's single-session verify, ~3.6 ms of wall a step for ~0.8 ms
        // of kernels. Anything below that must reach the driver (an allocation,
        // an upload, a readback) pauses the recording at its own call site and
        // resumes it.
        #[cfg(feature = "cuda")]
        let capture = match &self.model.device {
            Device::Cuda(cuda) => Some(cuda.begin_wave_capture()?),
            _ => None,
        };
        self.model.device.record_launches()?;
        let rewound = self.rewind_recorded(session, targets, &cap, &jobs);
        #[cfg(feature = "cuda")]
        let rewound = match (rewound, capture) {
            (Ok(()), Some(capture)) => capture.finish(),
            (rewound, _) => rewound,
        };
        // Consumed either way: its rows named this step's blocks and nothing
        // later. A sequence the fold reached is no longer pending; any armed
        // sequence it did not — one the step never reconciled, or every one when
        // the rewind failed — has its timelines dropped, so no pending layer
        // outlives the step.
        let released = self.release_capture(cap);
        rewound?;
        released
    }

    /// The device half of [`Self::rewind_cohort`]: the GDN fold, the PLE and
    /// QSA rewinds, the draft head's seed and the K/V truncation, for `jobs`
    /// (`(sequence, rows kept)`) of the blocks `cap` armed.
    fn rewind_recorded(
        &self,
        session: &mut BatchedInferenceSession,
        targets: &[(usize, usize)],
        cap: &SpecCapture,
        jobs: &[(usize, usize)],
    ) -> Result<()> {
        let cfg = &self.model.cfg;
        // ── The GDN half: one fold over the whole cohort, each sequence its
        // own kept rows — a full accept folds every row. The fold runs eagerly
        // inside the rewind's recording (`delta_net_fold`). ──
        {
            let mut rec = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("recurrent lock poisoned".into()))?;
            let mut stores: HashMap<usize, &mut RecurrentStateStore> = rec
                .iter_mut()
                .filter(|(seq, _)| jobs.iter().any(|&(s, _)| s == **seq))
                .map(|(seq, st)| (*seq, st))
                .collect();
            let mut folds: Vec<(&mut RecurrentStateStore, Option<usize>)> =
                Vec::with_capacity(jobs.len());
            for &(seq, kept) in jobs {
                let len = cap.seqs.get(&seq).map_or(0, |s| s.len);
                if kept > len {
                    candle::bail!(
                        "qwen4exp rewind: sequence {seq} keeps {kept} rows of a {len}-row block"
                    );
                }
                let store = stores.remove(&seq).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "qwen4exp rewind: sequence {seq} has no recurrent state"
                    ))
                })?;
                folds.push((store, Some(kept)));
            }
            fold_stores(&mut folds)?;
        }

        // ── The PLE and QSA halves: per sequence, bookkeeping over rows the
        // wave already computed. ──
        {
            let mut ple_map = self
                .ple
                .write()
                .map_err(|_| candle::Error::Msg("ple lock poisoned".into()))?;
            let mut idx_map = self
                .index
                .write()
                .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
            let ratios = self.attention_ratios();
            // In KV-layer order, the draft head's last — its cache advanced
            // over the same block the trunk's did, because it runs the same
            // rows in the same wave, so it rewinds with them.
            let mut indexer: Vec<&IndexerWeights> = self
                .model
                .layers
                .iter()
                .filter_map(|l| match &l.mix {
                    GpuLayerMix::Attention { indexer, .. } => Some(indexer),
                    GpuLayerMix::DeltaNet(_) => None,
                })
                .collect();
            if let Some(head) = &self.model.mtp {
                if let GpuLayerMix::Attention { indexer: i, .. } = &head.block.mix {
                    indexer.push(i);
                }
            }
            for &(seq, kept) in jobs {
                let Some(stash) = cap.seqs.get(&seq) else {
                    continue;
                };
                let (Some(ple), Some(caches)) = (ple_map.get_mut(&seq), idx_map.get_mut(&seq))
                else {
                    continue;
                };
                rewind_row_state(
                    stash,
                    kept,
                    ple,
                    caches,
                    &ratios,
                    &indexer,
                    cfg.ple.conv_history(),
                    cfg.ple.ngram_size,
                    cfg.rms_norm_eps,
                )?;
            }
        }

        // ── The draft head's seed. ──
        //
        // The lockstep pass stored the whole block's residual rows; the accept
        // walk decides which of them the next draft follows. Left at the
        // block's last row it would seed the head from a position the sequence
        // has just rolled back past — and the next lockstep pass would write
        // the head's *committed* K/V for the accepted position from a rejected
        // state, with nothing to raise but a decaying accept rate.
        {
            // Held exclusively while the seeds are rewritten on the device.
            let seeds = self
                .seeds
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            for &(seq, kept) in jobs {
                let Some(stash) = cap.seqs.get(&seq) else {
                    continue;
                };
                if kept == stash.len {
                    continue;
                }
                let Some(rows) = stash.head_rows.as_ref() else {
                    continue;
                };
                if kept == 0 || kept > stash.len {
                    candle::bail!(
                        "qwen4exp seed rewind: sequence {seq} accepted {kept} of a {}-row block",
                        stash.len
                    );
                }
                // Into the seed the block's carry wrote, which the rewind
                // replaces: the sequence keeps its two seed buffers.
                let seed = seeds.get(&seq).ok_or_else(|| {
                    candle::Error::Msg(format!("qwen4exp seed rewind: sequence {seq} has no seed"))
                })?;
                seed.slice_set(&rows.narrow(0, kept - 1, 1)?, 0, 0)?;
            }
        }

        // ── The K/V, which truncation does reach. ──
        for &(seq, tokens) in targets {
            if session.sequence_offset(seq) != Some(tokens) {
                session.truncate_sequence_to_tokens(seq, tokens)?;
            }
        }
        Ok(())
    }
}

/// Rewind one sequence's three recurrences to the `kept` rows its block had
/// accepted.
///
/// `kept` counts rows of the block, so `kept == len` is a full accept and does
/// nothing: the live state already covers exactly those tokens. The GDN half is
/// handled by the caller in one batched pass over the whole cohort — this is
/// the per-sequence bookkeeping the other two need.
// The rewind needs the captured rows, the cut, and then every piece of per-layer
// state it rebuilds — PLE, index caches, ratios, indexer weights — plus the
// window scalars. They are independent inputs to one operation, not a
// bundle that travels together anywhere else.
#[allow(clippy::too_many_arguments)]
pub fn rewind_row_state(
    stash: &SeqStash,
    kept: usize,
    ple: &mut PleState,
    caches: &mut [IndexCache],
    ratios: &[usize],
    indexer: &[&IndexerWeights],
    hist: usize,
    ngram: usize,
    eps: f64,
) -> Result<()> {
    if kept == stash.len {
        return Ok(());
    }
    if kept == 0 || kept > stash.len {
        candle::bail!(
            "qwen4exp verify rewind: {kept} accepted rows of a {}-row block — a block always \
             commits at least its first token, so zero is a bookkeeping fault rather than a \
             short accept",
            stash.len
        );
    }

    // ── PLE: the window the accepted rows would have left. ──
    //
    // `conv_hist` after `m` rows is the last `hist` rows of
    // `(entering_hist ++ block_rows)` — no arithmetic replayed, because the
    // rows are already computed and stashed. Written into the history the
    // block's wave left, which the rewind replaces wholesale: it is neither the
    // entering history (that is the spare now) nor the stashed rows, so the
    // copies read nothing they write, and the sequence keeps its two buffers.
    let entering = stash
        .ple_entering
        .as_ref()
        .ok_or_else(|| candle::Error::msg("qwen4exp verify rewind: no entering PLE state"))?;
    let dst = &ple.conv_hist;
    if dst.same_storage(&entering.conv_hist) {
        candle::bail!(
            "qwen4exp verify rewind: the PLE history to rewrite is the entering one — the \
             rewind would read the rows it writes"
        );
    }
    match &stash.ple_rows {
        Some(rows) if kept >= hist => {
            dst.slice_set(&rows.narrow(0, kept - hist, hist)?, 0, 0)?;
        }
        Some(rows) => {
            dst.slice_set(&entering.conv_hist.narrow(0, kept, hist - kept)?, 0, 0)?;
            dst.slice_set(&rows.narrow(0, 0, kept)?, 0, hist - kept)?;
        }
        // A wave whose window never reached the PLE layer captured no rows, and
        // then it advanced no history either: the entering one stands.
        None => dst.slice_set(&entering.conv_hist, 0, 0)?,
    }
    // The hash window is a function of token ids alone.
    let mut prev = entering.prev.clone();
    prev.extend(stash.tokens.iter().take(kept).copied());
    let keep = ngram.saturating_sub(1);
    if prev.len() > keep {
        prev.drain(..prev.len() - keep);
    }
    ple.prev = prev;

    // ── QSA: back to the entering cache, then re-append the accepted rows. ──
    //
    // Through `append_wave`, the same call the wave made, rather than a second
    // transcription of the pooling — a cache rebuilt by different arithmetic
    // than the one it replaces is a selection that drifts from the wave's. One
    // span, because a rewind restores one sequence.
    // **The restore is not optional, and appending without it is the bug it
    // used to hide.** The wave advanced this cache over the WHOLE block; the
    // re-append then adds the accepted rows on top. Those two are only correct
    // as a pair — restore first, then append — because the append assumes it is
    // building on the state the block started from. Skip the restore and the
    // cache keeps the block's rows *and* gains the accepted ones, so the index
    // ends exactly `block` tokens past the K/V.
    //
    // Measured before this refused: every speculative step left the index 5
    // ahead of a 5-token block, on a full accept as much as a partial one, and
    // the surplus rode into the seal as pages covering tokens the turn does not
    // hold. `qsa_entering` was empty, `get(kv)` returned `None`, and the `if let`
    // simply moved on. A missing snapshot is a bug in whoever armed the capture,
    // never something to continue past.
    let n_caches = caches.len();
    for (kv, cache) in caches.iter_mut().enumerate() {
        let snap = stash.qsa_entering.get(kv).ok_or_else(|| {
            candle::Error::Msg(format!(
                "qwen4exp rewind: no entering index snapshot for KV layer {kv} (the capture \
                 armed {} of {n_caches} layers) — the wave advanced this cache over the whole \
                 block and appending the accepted rows without restoring first would leave it \
                 a block past the K/V",
                stash.qsa_entering.len(),
            ))
        })?;
        cache.restore(snap)?;
        let ratio = ratios.get(kv).copied().unwrap_or(0);
        if ratio == 0 {
            continue;
        }
        if let (Some(Some(keys)), Some(w)) = (stash.qsa_keys.get(kv), indexer.get(kv)) {
            let mut one = [AppendSpan {
                cache,
                start: 0,
                rows: kept,
            }];
            // Replay of a captured span, outside any open layer phase.
            append_wave(&mut one, keys, w, ratio, eps)?;
        }
    }
    Ok(())
}
