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
//! | GDN `S` (DeltaNet) | a running sum over every token, no per-token decomposition | replay the mixer over the accepted rows from the entering state |
//! | PLE conv history | a sliding window of derived rows, not of tokens | slice the window that the accepted rows would have left |
//! | QSA index cache | pooled block keys, appended as blocks complete | restore the entering snapshot, re-append the accepted rows' keys |
//!
//! # The way back is forward, and the entering state is already free
//!
//! Each of the three is snapshotted at wave entry *anyway*, for the failure
//! bracket that rolls a failed wave back ([`super::wave`]). A verify wave keeps
//! those snapshots instead of dropping them, so the "state the block was
//! entered with" costs nothing extra. What a rewind adds is the block's own
//! per-row operands, because the wave arena reclaims them when the forward
//! ends — so a verifying span stashes them as it goes.
//!
//! Only the GDN half replays arithmetic; it runs through
//! [`replay_accepted_prefixes`], the same function the hybrid runs, against the
//! same [`VerifyStash`]. The other two are bookkeeping over rows already
//! computed: the PLE window is a slice, and the index cache re-appends through
//! [`IndexCache::append`] — the same call the wave made.
//!
//! # Cost
//!
//! For a block of `k` proposals accepted at `m`: one mixer call per DeltaNet
//! layer over `m ≤ k+1` rows, one narrow, and one `append` per attention layer,
//! against a whole forward's 48 layers of projections, attention and a
//! 512-expert MoE.

use std::collections::HashMap;

use candle::{Result, Tensor};

use super::engine::GpuLayerMix;
use super::indexer::{append_wave, AppendSpan, IndexCache, IndexSnapshot};
use super::ple::PleState;
use super::qsa::IndexerWeights;
use super::wave::Qwen4ExpBatched;
use crate::models::batched_inference::{BatchedInferenceSession, ManagedBatchedModel};
use crate::models::delta_net::{
    DeltaNetConstants, DeltaNetDims, LayerKind, RecurrentStateStore, SpanOperands,
};
use crate::models::qwen35::attention::RopeTables;
use crate::models::qwen35::spec::{replay_accepted_prefixes, ReplayLayer, StashSpan, VerifyStash};
use candle_nn::kv_cache::vram_budget_available;

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
    /// Per verifying sequence.
    pub seqs: HashMap<usize, SeqStash>,
    /// The cohort's GDN operands, one buffer set per recurrent layer.
    pub delta: VerifyStash,
}

impl SpecCapture {
    /// Arm for `blocks` — `(sequence, block length)`, in wave order.
    pub fn new(
        blocks: &[(usize, usize)],
        layer_kinds: &[LayerKind],
        dims: &DeltaNetDims,
        device: &candle::Device,
    ) -> Result<Self> {
        let cap: usize = blocks.iter().map(|&(_, n)| n).sum();
        let mut delta = VerifyStash::new(layer_kinds, dims, cap, device)?;
        delta.begin(blocks)?;
        let mut seqs = HashMap::with_capacity(blocks.len());
        for &(seq, len) in blocks {
            let span = delta.span_of(seq).ok_or_else(|| {
                candle::Error::Msg(format!("qwen4exp verify: no stash span for sequence {seq}"))
            })?;
            seqs.insert(
                seq,
                SeqStash {
                    row: span.row,
                    len,
                    ..SeqStash::default()
                },
            );
        }
        Ok(Self { seqs, delta })
    }

    /// The GDN stash rows for `seq`, or `None` when it is not verifying.
    pub fn delta_span(&self, seq: usize) -> Option<StashSpan> {
        self.delta.span_of(seq)
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

    /// A GDN layer's operand buffers, for a sweep that is about to run the
    /// mixer over this cohort.
    pub fn delta_layer(&mut self, ord: usize) -> Option<&mut SpanOperands> {
        self.delta.layers.get_mut(ord)
    }
}

impl Qwen4ExpBatched {
    /// The deepest budget whose rewind stash the KV side can currently hold.
    ///
    /// `usize::MAX` when the question does not arise — no reservation to
    /// measure (a CPU device, or a test). The caller `min`s with the ladder, so
    /// an unmeasurable bound never *raises* a budget.
    pub(super) fn affordable_draft_budget(&self, width: usize) -> usize {
        let dims = &self.model.cfg.delta_net;
        let layers = self
            .model
            .cfg
            .layer_kinds
            .iter()
            .filter(|k| matches!(k, LayerKind::DeltaNet))
            .count();
        // The four operands `SpanOperands` holds, per row, per DeltaNet layer.
        let per_row = (dims.conv_dim() + dims.value_dim() + 2 * dims.n_v_heads)
            * std::mem::size_of::<f32>()
            * layers;
        if per_row == 0 || width == 0 {
            return usize::MAX;
        }
        let Some(budget) = vram_budget_available(&self.model.device) else {
            return usize::MAX;
        };
        // Half of what is claimable, not all of it: the stash is one tenant
        // among several, and a wave that spends every free region on its own
        // rewind buffer has nothing left to decode into.
        let rows = (budget / 2) / per_row;
        (rows / width).saturating_sub(1)
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
        let cfg = &self.model.cfg;
        let mut guard = self
            .verify
            .write()
            .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?;

        // **Taken, not borrowed.** A stash span is good for exactly one step,
        // and a failed replay's must not survive to rewind a later one. The
        // borrow left the capture armed on every `?` below, so a cohort that
        // bailed partway — some sequences rewound, none truncated — kept a
        // stash whose spans still read as a valid rewind point for the next
        // `truncate_sequences`. Consuming it up front makes a later rewind
        // refuse outright, which is the honest report of a step that failed.
        // qwen35 consumes unconditionally for the same reason (`qwen35/forward.rs`).
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
        for &(seq, tokens) in targets {
            let Some(stash) = cap.seqs.get(&seq) else {
                // A plain decode row: it wrote one token and kept it.
                if session.sequence_offset(seq) != Some(tokens) {
                    candle::bail!(
                        "qwen4exp rewind: sequence {seq} did not verify a block this step but \
                         is asked to move to {tokens} tokens"
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
                    "qwen4exp rewind: sequence {seq} asked to move to {tokens} tokens, before \
                     the block this capture covers (starts at {base}) — a replay covers the \
                     block it stashed operands for and nothing else"
                ))
            })?;
            jobs.push((seq, kept));
        }

        // ── The GDN half: one batched replay over the whole cohort. ──
        let short: Vec<(usize, usize)> = jobs
            .iter()
            .copied()
            .filter(|&(seq, kept)| cap.seqs.get(&seq).is_some_and(|s| kept < s.len))
            .collect();
        if !short.is_empty() {
            let recurrent: Vec<usize> = cfg
                .layer_kinds
                .iter()
                .enumerate()
                .filter(|(_, k)| matches!(k, LayerKind::DeltaNet))
                .map(|(i, _)| i)
                .collect();
            let layers: Vec<ReplayLayer<'_>> = recurrent
                .iter()
                .map(|&li| match &self.model.layers[li].mix {
                    GpuLayerMix::DeltaNet(w) => Ok(ReplayLayer {
                        layer_index: li,
                        consts: DeltaNetConstants {
                            dt_bias: &w.dt_bias,
                            a: &w.a,
                            conv: &w.conv,
                            norm: &w.norm,
                        },
                    }),
                    GpuLayerMix::Attention { .. } => candle::bail!(
                        "qwen4exp rewind: layer {li} carries recurrent state but is not DeltaNet"
                    ),
                })
                .collect::<Result<_>>()?;

            let mut rec = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("recurrent lock poisoned".into()))?;
            let mut stores: HashMap<usize, &mut RecurrentStateStore> = rec
                .iter_mut()
                .filter(|(seq, _)| short.iter().any(|&(s, _)| s == **seq))
                .map(|(seq, st)| (*seq, st))
                .collect();
            let mut full: Vec<(StashSpan, usize, &mut RecurrentStateStore)> =
                Vec::with_capacity(short.len());
            for &(seq, kept) in &short {
                let span = cap.delta.span_of(seq).ok_or_else(|| {
                    candle::Error::Msg(format!("qwen4exp rewind: no stash span for {seq}"))
                })?;
                let store = stores.remove(&seq).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "qwen4exp rewind: sequence {seq} has no recurrent state"
                    ))
                })?;
                full.push((span, kept, store));
            }
            replay_accepted_prefixes(
                &layers,
                &cfg.delta_net,
                cfg.rms_norm_eps,
                &self.model.device,
                |d| self.wave_geometry(d),
                &cap.delta,
                &mut full,
            )?;
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
            let depth = idx_map
                .values()
                .flat_map(|c| c.iter())
                .map(|c| c.capacity_blocks())
                .max()
                .unwrap_or(0);
            let rope = self.index_rope_for(depth)?;
            for &(seq, kept) in &jobs {
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
                    &rope,
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
            let mut seeds = self
                .seeds
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            for &(seq, kept) in &jobs {
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
                seeds.insert(seq, rows.narrow(0, kept - 1, 1)?.to_owned_tensor()?);
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
// The rewind needs the stash, the cut, and then every piece of per-layer state
// the replay has to advance again — PLE, index caches, ratios, indexer weights,
// rope tables — plus the two window scalars. They are independent inputs to one
// operation, not a bundle that travels together anywhere else.
#[allow(clippy::too_many_arguments)]
pub fn rewind_row_state(
    stash: &SeqStash,
    kept: usize,
    ple: &mut PleState,
    caches: &mut [IndexCache],
    ratios: &[usize],
    indexer: &[&IndexerWeights],
    rope: &RopeTables,
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
    // `(entering_hist ++ block_rows)`, so it is `narrow(m, hist)` over that
    // concatenation — no arithmetic replayed, because the rows are already
    // computed and stashed.
    let entering = stash
        .ple_entering
        .as_ref()
        .ok_or_else(|| candle::Error::msg("qwen4exp verify rewind: no entering PLE state"))?;
    match &stash.ple_rows {
        Some(rows) => {
            let padded = Tensor::cat(&[&entering.conv_hist, rows], 0)?;
            // Owned for the same reason `ple_apply` owns its own tail: this is
            // carried state, and a `contiguous()` narrow would alias `padded`
            // and pin the whole concatenation alive to keep `hist` rows.
            ple.conv_hist = padded.narrow(0, kept, hist)?.to_owned_tensor()?;
        }
        // A wave whose window never reached the PLE layer captured no rows, and
        // then it advanced no history either.
        None => ple.conv_hist = entering.conv_hist.clone(),
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
            append_wave(&mut one, keys, w, rope, ratio, eps)?;
        }
    }
    Ok(())
}
