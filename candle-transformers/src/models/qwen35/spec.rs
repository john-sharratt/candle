//! Speculative verify for the hybrid: rewinding a recurrence that has no
//! suffix to remove.
//!
//! A speculative step runs a block of proposed tokens through one forward and
//! then learns how many of them the model actually agrees with. For the
//! attention half that is free — paged KV is append-only, so a truncation to
//! the accepted length erases exactly the rejected tokens. For the DeltaNet
//! half it is not: `S` is a running sum over every token of the sequence, with
//! no per-token decomposition, which is why
//! [`ManagedBatchedModel::truncate_sequence`](crate::models::batched_inference::ManagedBatchedModel::truncate_sequence)
//! on this model used to refuse any non-zero rewind outright.
//!
//! The way back is forward. Two facts compose:
//!
//! * The store's ping-pong means a wave writes the buffer it is *not* reading,
//!   so immediately after `commit_wave` the non-live half still holds the state
//!   the block was entered with — untouched, at no cost
//!   ([`RecurrentStateStore::layer_state_rewind`]).
//! * The mixer's arithmetic for row `i` depends on row `i` and the rows before
//!   it, and on nothing after. So re-running the mixer over the block's first
//!   `m` rows, from that entering state, produces exactly the state the model
//!   would have had if only those `m` tokens had ever been decoded.
//!
//! What the replay needs is the block's *operands*, which the wave arena
//! reclaims when the forward ends — so a verifying span stashes them as it goes
//! ([`SpanOperands`]), one set per DeltaNet layer. Post-projection deliberately:
//! re-running the projections would be re-deriving numbers whose bit-identity
//! rests on a GEMM's reduction order not depending on its row count, and the
//! whole point of the rewind is that the sequence cannot tell speculation
//! happened.
//!
//! The replay runs through [`delta_net_mix_spans`] — the same function the wave
//! ran, not a second transcription of it — so it takes whichever path the wave
//! took (fused CUDA kernels or the tensor-op reference) and matches it by
//! construction. Its output activations are discarded; only the advanced state
//! is wanted.
//!
//! Cost, for a block of `k` proposals accepted at `m`: `n_deltanet_layers`
//! mixer calls over `m ≤ k+1` rows, against a whole forward's 48 layers of
//! projections, attention, and a 512-expert MoE. On the measured hybrid that is
//! a few percent of the wave it lets us skip.

use candle::{Device, Result, Tensor};

use crate::models::delta_net::{
    delta_net_advance_spans, DeltaNetConstants, DeltaNetDims, DeltaNetOut, DeltaNetSeq,
    DeltaNetState, LayerKind, RecurrentStateStore, SpanOperands,
};

use super::quantized_weights::{QuantLayerMix, QuantModel};

/// The COHORT's stashed speculative blocks: every verifying sequence's rows in
/// one set of shared buffers, so the replay that consumes them advances every
/// sequence's state in one batched launch per layer.
///
/// `layers` is in sweep order — the same order
/// [`RecurrentStateStore::recurrent_layer_indices`] yields, because both walk
/// the trunk forwards — so entry `j` belongs to the `j`-th recurrent layer.
/// `spans` records which rows belong to which sequence, in the wave's own
/// spec-span order, so the row ranges ascend and never overlap.
pub struct VerifyStash {
    /// Per recurrent layer, in sweep order; each holds the whole cohort.
    pub layers: Vec<SpanOperands>,
    /// Per verifying sequence.
    pub spans: Vec<StashSpan>,
}

/// One sequence's rows within the cohort stash.
#[derive(Debug, Clone, Copy)]
pub struct StashSpan {
    pub seq: usize,
    /// First row in the shared buffers.
    pub row: usize,
    /// Absolute position of the block's first token, set by the sweep that
    /// filled the buffers.
    pub start: usize,
    /// Rows the sweep captured for this sequence.
    pub len: usize,
}

impl VerifyStash {
    /// Buffers for a cohort of up to `cap` verify rows across every recurrent
    /// layer of `layer_kinds`.
    ///
    /// **Allocate outside a forward.** A wave's storage is claimed before the
    /// forward opens and the transient tier is placed against that claim, so a
    /// device allocation from inside it is refused — which is exactly what a
    /// stash that allocated as the sweep passed each layer would be. The
    /// buffers are sized for the widest cohort the caller will verify and
    /// reused across steps.
    pub fn new(
        layer_kinds: &[LayerKind],
        dims: &DeltaNetDims,
        cap: usize,
        dev: &Device,
    ) -> Result<Self> {
        let n = layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::DeltaNet)
            .count();
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            layers.push(SpanOperands::zeros(dims, cap, dev)?);
        }
        Ok(Self {
            layers,
            spans: Vec::new(),
        })
    }

    /// Rows these buffers can hold.
    pub fn capacity(&self) -> Result<usize> {
        match self.layers.first() {
            Some(l) => l.capacity(),
            None => Ok(0),
        }
    }

    /// Lay out this step's cohort: one span per verifying sequence, rows packed
    /// in the given order. Replaces whatever the previous step left.
    ///
    /// The caller must have sized the buffers first — this only records where
    /// each sequence's rows will land, and refuses a cohort the buffers cannot
    /// hold rather than letting the wave capture past them.
    pub fn begin(&mut self, blocks: &[(usize, usize)]) -> Result<()> {
        let total: usize = blocks.iter().map(|&(_, len)| len).sum();
        let cap = self.capacity()?;
        if total > cap {
            candle::bail!("qwen35 verify stash: a {total}-row cohort against {cap}-row buffers");
        }
        self.spans.clear();
        let mut row = 0usize;
        for &(seq, len) in blocks {
            self.spans.push(StashSpan {
                seq,
                row,
                start: 0,
                len,
            });
            row += len;
        }
        Ok(())
    }

    /// This sequence's span, if the last verify wave stashed one for it.
    pub fn span_of(&self, seq: usize) -> Option<StashSpan> {
        self.spans.iter().copied().find(|s| s.seq == seq)
    }

    /// Drop a sequence's span — after its replay, or to invalidate it. The
    /// buffers stay; a stash span is good for exactly one rewind, and a second
    /// use would replay from a state two waves old.
    pub fn remove(&mut self, seq: usize) {
        self.spans.retain(|s| s.seq != seq);
    }
}

/// Re-advance every job's store from the state it entered its stashed block
/// with to the state after the block's first `kept` tokens — the whole cohort
/// in one batched launch pair per recurrent layer, through the same span-table
/// kernels the verify wave itself ran.
///
/// Call **once per step**, immediately after the verify wave committed and
/// before any other wave touches these sequences: the entering states live in
/// each store's non-live half only until the next wave writes there.
///
/// A job whose `kept == span.len` is a full accept and is skipped without
/// touching anything — its live state already covers exactly those tokens.
pub fn replay_accepted_prefixes(
    model: &QuantModel,
    stash: &VerifyStash,
    jobs: &mut [(StashSpan, usize, &mut RecurrentStateStore)],
) -> Result<()> {
    for (span, kept, _) in jobs.iter() {
        if *kept == 0 {
            candle::bail!(
                "qwen35 verify replay: a block always commits at least its first token, \
                 so a rewind to zero rows is a bookkeeping fault, not a short accept"
            );
        }
        if *kept > span.len {
            candle::bail!(
                "qwen35 verify replay: {kept} accepted rows of a {}-row block",
                span.len
            );
        }
    }
    // Full accepts need nothing — the live state already covers exactly their
    // tokens. What remains ascends by stash row, because the spans were laid
    // out in wave order and a filter keeps order.
    let mut short: Vec<&mut (StashSpan, usize, &mut RecurrentStateStore)> = jobs
        .iter_mut()
        .filter(|(span, kept, _)| *kept < span.len)
        .collect();
    if short.is_empty() {
        return Ok(());
    }
    let layer_indices: Vec<usize> = short[0].2.recurrent_layer_indices().collect();
    if stash.layers.len() != layer_indices.len() {
        candle::bail!(
            "qwen35 verify replay: {} stashed layers against {} recurrent layers — the \
             verify wave did not stash every DeltaNet layer it swept",
            stash.layers.len(),
            layer_indices.len()
        );
    }
    let dims: &DeltaNetDims = &model.cfg.delta_net;
    let eps = model.cfg.rms_norm_eps;
    for (ord, &li) in layer_indices.iter().enumerate() {
        let QuantLayerMix::DeltaNet(w) = &model.layers[li].mix else {
            candle::bail!(
                "qwen35 verify replay: layer {li} carries recurrent state but is not DeltaNet"
            );
        };
        let p = stash.layers[ord].all_rows();
        let c = DeltaNetConstants {
            dt_bias: &w.dt_bias,
            a: &w.a,
            conv: &w.conv,
            norm: &w.norm,
        };
        // One span per rewinding sequence, over its own rows of the shared
        // buffers. For each: READ the half the block was entered from, WRITE
        // the live one — the shorter advance replaces the block-length advance
        // in place, and the half being read is about to become the next wave's
        // write buffer, so whatever the replay leaves in it does not survive.
        let mut seqs: Vec<DeltaNetSeq<'_>> = Vec::with_capacity(short.len());
        for (span, kept, store) in short.iter_mut() {
            let (entering, out): (&mut DeltaNetState, DeltaNetOut) =
                store.layer_state_rewind(li)?;
            seqs.push(DeltaNetSeq {
                start: span.row,
                len: *kept,
                state: entering,
                out,
                stash: None,
            });
        }
        // The gated activations are the layer's output, which the accepted
        // tokens' logits were already produced from. Only the states are
        // wanted, and every sequence's advances in ONE launch pair.
        delta_net_advance_spans(&p, &c, dims, &mut seqs, eps)?;
    }
    Ok(())
}

/// The rows of `logits` a verify block scored, split off a wave's output.
///
/// The head emits one row per scored position in wave order, and a verify
/// wave's order is `[plain decode rows | each block's rows]` — this is the
/// split, kept beside the replay because the two are the same bookkeeping seen
/// from either end.
pub fn split_block_rows(
    logits: &[Tensor],
    n_plain: usize,
    block_lens: &[usize],
) -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)> {
    let want: usize = n_plain + block_lens.iter().sum::<usize>();
    if logits.len() != want {
        candle::bail!(
            "qwen35 verify: wave scored {} rows, expected {want} ({n_plain} plain + \
             blocks {block_lens:?}) — the head did not score every verify row",
            logits.len()
        );
    }
    let plain = logits[..n_plain].to_vec();
    let mut blocks = Vec::with_capacity(block_lens.len());
    let mut off = n_plain;
    for &l in block_lens {
        blocks.push(logits[off..off + l].to_vec());
        off += l;
    }
    Ok((plain, blocks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn row(v: f32) -> Tensor {
        Tensor::full(v, (1, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    }

    #[test]
    fn block_rows_split_after_the_plain_prefix() {
        let rows: Vec<Tensor> = (0..6).map(|i| row(i as f32)).collect();
        let (plain, blocks) = split_block_rows(&rows, 2, &[3, 1]).unwrap();
        assert_eq!(plain.len(), 2);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].len(), 3);
        assert_eq!(blocks[1].len(), 1);
        let first = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert_eq!(first(&plain[0]), 0.0);
        assert_eq!(first(&blocks[0][0]), 2.0);
        assert_eq!(first(&blocks[1][0]), 5.0);
    }

    /// A short count is the symptom of the head scoring only each prefill
    /// span's LAST row, which is what it does on an ordinary wave — so it must
    /// be an error here, not a silent misalignment of every block's argmaxes.
    #[test]
    fn a_short_row_count_is_refused() {
        let rows: Vec<Tensor> = (0..3).map(|i| row(i as f32)).collect();
        let err = split_block_rows(&rows, 1, &[3]).unwrap_err().to_string();
        assert!(err.contains("score every verify row"), "{err}");
    }
}
