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

#[cfg(feature = "cuda")]
use crate::models::delta_net::state_store::RegionBump;
use crate::models::delta_net::{
    delta_net_advance_spans, DeltaNetConstants, DeltaNetDims, DeltaNetOut, DeltaNetProjections,
    DeltaNetSeq, DeltaNetState, LayerKind, RecurrentStateStore, SpanOperands,
};
#[cfg(feature = "cuda")]
use crate::models::wave_buffers::wave_empty;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{begin_wave, LayerPhase, WaveGeneration};

use super::quantized_weights::QuantModel;

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
    /// Which recurrent layers this cohort's sweep has actually captured, by the
    /// same ordinal that indexes `layers`.
    ///
    /// A sweep split into layer windows fills its own ordinals and leaves the
    /// rest to the window that follows, so the buffers being *allocated* says
    /// nothing about whether they were *written* — and a replay from a
    /// half-written stash advances some layers and not others, silently. This
    /// is the record that makes the difference checkable.
    pub filled: Vec<bool>,
    /// The reservation regions `layers` is carved from.
    ///
    /// **The regions, not the allocator that claimed them.** Keeping the
    /// `RegionBump` would keep its `SpanClaims` alive, and that is an open arena
    /// window — every later wave blocks in `wave_gate` waiting for it to close.
    /// Measured as a 58-minute hang with the process alive and not one line of
    /// output. `RegionBump::into_regions` is the handover.
    ///
    /// **Declared last, and that is load-bearing.** Struct fields drop in
    /// declaration order, so `layers` — whose tensors are `Foreign` leases
    /// pointing into these regions — must be gone before the regions return to
    /// the free list. Moving this field up would leave every buffer above it
    /// naming ground another claimant may already hold.
    ///
    /// Empty on a device with no reservation to carve from: a CPU device, or a
    /// unit test. Those fall back to driver memory, which is what the whole
    /// stash used to do.
    ///
    /// Never read: it is an RAII holder and dropping it is the whole of its job.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    regions: Vec<candle_nn::kv_cache::SpanRegion>,
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
        // From the reservation where there is one, so the stash trades against
        // KV and weights like every other long-lived buffer instead of
        // competing invisibly for the card outside the span. See
        // [`SpanOperands::in_regions`] for the measurement that made this
        // necessary.
        #[cfg(feature = "cuda")]
        let mut regions = RegionBump::for_device(dev)?;
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            #[cfg(feature = "cuda")]
            let ops = match regions.as_mut() {
                Some(bump) => SpanOperands::in_regions(dims, cap, dev, bump)?,
                None => SpanOperands::zeros(dims, cap, dev)?,
            };
            #[cfg(not(feature = "cuda"))]
            let ops = SpanOperands::zeros(dims, cap, dev)?;
            layers.push(ops);
        }
        Ok(Self {
            layers,
            spans: Vec::new(),
            filled: vec![false; n],
            // Takes the regions and drops the bump, closing the arena window
            // before this returns — see the field's own note.
            #[cfg(feature = "cuda")]
            regions: regions.map_or_else(Vec::new, RegionBump::into_regions),
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
        // A new cohort has captured nothing yet, whatever the last one left.
        self.filled.iter_mut().for_each(|f| *f = false);
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

    /// Whether any sequence still names a span in this stash.
    ///
    /// **The buffers outlive a span deliberately and must not outlive every
    /// span.** Keeping them across steps is the point — they are reallocated
    /// only when a wider cohort arrives — but once no sequence names one, the
    /// stash is holding reservation regions on behalf of nobody, and it holds
    /// them for the life of the process. It is not KV, so no arena sweep sees
    /// it and every KV-side diagnostic reports the pool as healthy.
    pub fn is_unused(&self) -> bool {
        self.spans.is_empty()
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
/// One layer's stashed operands, copied onto the wave's half.
///
/// The **provenance root** for a replay. Every buffer the mixer allocates is
/// placed beside one of these, so leasing them here is what keeps the whole
/// chain off the pool — see [`replay_accepted_prefixes`] for the measurement
/// that made it necessary.
///
/// Without a wave (no CUDA, or a caller that could not open a generation) this
/// hands back the stash's own tensors unchanged: the replay is still correct,
/// it simply allocates the way it always did.
#[cfg(feature = "cuda")]
fn stage_on_wave<'w>(
    ops: &SpanOperands,
    device: &Device,
    wave: Option<&'w WaveGeneration>,
) -> Result<DeltaNetProjections<'w>> {
    let Some(_) = wave else {
        return Ok(ops.all_rows());
    };
    // Uninitialised, not zeroed: the `slice_set` below writes every element, and
    // a `memset` first would be a full-width pass per operand per layer that
    // nothing reads (hot-path invariant 6).
    let stage = |src: &Tensor| -> Result<candle::LiveTensor<'w>> {
        let dst = wave_empty(src.shape(), src.dtype(), device, wave)?;
        dst.slice_set(src, 0, 0)?;
        Ok(dst)
    };
    Ok(DeltaNetProjections {
        qkv: stage(&ops.qkv)?,
        z: stage(&ops.z)?,
        beta_lin: stage(&ops.beta_lin)?,
        alpha_lin: stage(&ops.alpha_lin)?,
    })
}

#[cfg(not(feature = "cuda"))]
fn stage_on_wave<'w>(
    ops: &SpanOperands,
    _device: &Device,
    _wave: Option<&'w ()>,
) -> Result<DeltaNetProjections<'static>> {
    Ok(ops.all_rows())
}

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
    // Allocated is not written. A sweep split into layer windows fills the
    // ordinals of the window it ran, and the windows accumulate into one stash
    // — so a missing ordinal here means some window never ran, and replaying
    // would advance the layers that were captured while leaving the rest at the
    // block's entering state.
    if let Some(ord) = stash.filled.iter().position(|f| !f) {
        candle::bail!(
            "qwen35 verify replay: recurrent layer {ord} of {} was never captured — the \
             verify's sweep did not cover every DeltaNet layer, so a rewind would advance \
             some layers and not others",
            stash.filled.len(),
        );
    }
    let dims: &DeltaNetDims = &model.cfg.delta_net;
    let eps = model.cfg.rms_norm_eps;

    // **A generation for the replay, because the stash has no provenance to
    // lend.**
    //
    // `SpanOperands` is allocated with `Tensor::zeros` outside any forward — the
    // sequence owns it across waves, which is the whole point of a rewind stash —
    // so its tensors are `Owned` and name no arena. The mixer then builds every
    // intermediate with `empty_beside`, and beside an `Owned` operand is the
    // pool: `conved`, then `u`/`w`/`kq`/`g_cs`, then everything downstream, per
    // DeltaNet layer, per rewinding sequence, on every accept.
    //
    // Measured with `--features forbidden_allocations` on the 27B: **20.0 GB** of
    // driver allocation at 20 contexts against 921 MB at one, on a card with
    // ~258 MiB outside the reservation. It surfaced as
    // `CUDA_ERROR_OUT_OF_MEMORY` on an unrelated event record, because by then
    // the device was simply full — and no region-pool diagnostic showed distress,
    // since none of it went through the pool.
    //
    // Speculation is what made it reachable at that scale: a rewind happens
    // exactly when proposals are rejected, so enabling drafting at width 20
    // multiplied this path by the cohort.
    //
    // Opening a generation is not sufficient on its own — `empty_beside` relays
    // provenance rather than creating it, so the *root* must be leased. Hence the
    // staging copy below.
    //
    // **The generation is per layer, not per replay.** The span is sized for one
    // layer's attention phase; holding one guard across the sweep accumulates
    // every layer's staging in it and exhausts it — measured, at layer 48 of the
    // first config: *"transient span exhausted — 491520 B at offset 23240704
    // exceeds the 23638784 B budget"*. Dropping the guard each iteration returns
    // the staging **and** the mixer's own intermediates before the next layer
    // asks, which is the same lifetime a forward gives its phases.
    for (ord, &li) in layer_indices.iter().enumerate() {
        // The **residue**, not the layer. The replay runs at accept time, well
        // after the sweep that captured the stash, so on a streamed checkpoint
        // this layer may long since have been evicted — and `ensure`ing it would
        // pull ~240 MB over PCIe to read four small constants that never left
        // VRAM. The residue holds exactly those four.
        let residue = model.layers.residue(li)?;
        let w = residue.delta_net().map_err(|_| {
            candle::Error::Msg(format!(
                "qwen35 verify replay: layer {li} carries recurrent state but is not DeltaNet"
            ))
        })?;
        #[cfg(feature = "cuda")]
        let wave: Option<WaveGeneration> = match &model.device {
            Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?),
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let wave: Option<()> = None;
        // The stash staged onto the wave's half, so the chain the mixer builds
        // from it has a leased root. Four copies per layer of buffers the
        // capture already copied once — against the pool traffic above, and
        // against a `contiguous()` the `rows()` path would have paid anyway.
        let p = stage_on_wave(&stash.layers[ord], &model.device, wave.as_ref())?;
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
