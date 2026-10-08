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
//! The replay runs the same arithmetic the wave ran, not a second transcription
//! of it: on the device the very prefill kernels the wave launched
//! (`delta_net::replay_stack`), elsewhere the tensor-op reference the kernels
//! are parity-locked to. Its output activations are discarded; only the
//! advanced state is wanted.
//!
//! Cost, for a block of `k` proposals accepted at `m`: the mixer over `m ≤ k+1`
//! rows in every DeltaNet layer — on the device one launch triple for a stack of
//! layers, since no layer's replay reads another's — against a whole forward's
//! 48 layers of projections, attention, and a 512-expert MoE.

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use candle::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
use crate::models::delta_net::cuda::DELTA_NET_PREFILL_DIM;
#[cfg(feature = "cuda")]
use crate::models::delta_net::replay_stack::{
    delta_net_replay_stack, ReplaySpan, ReplayStates, StackedLayer,
};
#[cfg(feature = "cuda")]
use crate::models::delta_net::DeltaNetProjections;
use crate::models::delta_net::{
    delta_net_advance_spans, DeltaNetConstants, DeltaNetDims, DeltaNetOut, DeltaNetSeq,
    DeltaNetState, LayerKind, RecurrentStateStore, SpanOperands,
};
#[cfg(feature = "cuda")]
use crate::models::wave_buffers::wave_empty;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{
    begin_wave, claim_arena_slots, plan_slot_moves, slot_stride, ArenaSlot, DeltaNetWidths,
    LayerPhase, SlotTenant, WaveGeneration,
};

/// The COHORT's stashed speculative blocks: every verifying sequence's rows in
/// one set of shared buffers, so the replay that consumes them advances every
/// sequence's state, in a stack of layers, in one batched launch.
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
    /// Spans the last [`Self::begin`] laid out. [`Self::remove`] leaves it
    /// alone: it is the cohort the verify forward priced the replay's carves
    /// from, and a rewind removes the spans it consumes before replaying them.
    cohort: usize,
    /// Which recurrent layers this cohort's sweep has actually captured, by the
    /// same ordinal that indexes `layers`.
    ///
    /// A sweep split into layer windows fills its own ordinals and leaves the
    /// rest to the window that follows, so the buffers being *allocated* says
    /// nothing about whether they were *written* — and a replay from a
    /// half-written stash advances some layers and not others, silently. This
    /// is the record that makes the difference checkable.
    pub filled: Vec<bool>,
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

/// Pack the rewind stash's arenas — the two-cursor pass, per stride.
///
/// **A stash outlives forwards but not the cap it was built for.** Its slot stride
/// is `cap × width × 4`, so every cohort width that has ever been verified opened
/// its own pools; the arenas of the caps that came before do not disappear when a
/// wider stash replaces them, they empty, and whatever is still live in them sits
/// where the earlier cohorts left it. Over a long run that is exactly the scatter
/// this packs.
///
/// **Outstanding spans do not block it, and gating on them made it dead code.** A
/// stash with no spans never exists to be packed: `SpecCapture::new` lays the
/// cohort out with `begin` before anything else can reach it, and `rewind_cohort`
/// disarms by taking the whole capture rather than by clearing spans — so a gate
/// of `is_unused()` is false for the stash's entire life and true only when there
/// is no stash. What actually makes a move safe is *when* this runs: captures are
/// written during a forward, replays resolve their addresses through
/// `SpanOperands::rows` at the moment they run, and both they and this pass are on
/// the scheduler's own thread between forwards. There is no window in which a
/// buffer both moves and is read.
///
/// The widths are deduplicated: `beta_lin` and `alpha_lin` are both `n_v_heads`
/// wide, so they share a pool, and planning the same stride twice would have the
/// second walk treat the first's claimed destinations as occupied ground.
#[cfg(feature = "cuda")]
pub fn compact_verify_stash(
    stash: &mut VerifyStash,
    dims: &DeltaNetDims,
    device: &Device,
    max_moves: usize,
) -> Result<(usize, usize)> {
    if !matches!(device, Device::Cuda(_)) {
        return Ok((0, 0));
    }
    let cap = stash.capacity()?;
    if cap == 0 {
        return Ok((0, 0));
    }
    let f32_bytes = DType::F32.size_in_bytes();
    // Deduplicated on the STRIDE the arena will actually use, not on the byte
    // count: two widths that differ but round up to the same 256-aligned stride
    // share one pool, and planning that pool twice would have the second walk read
    // the first's claimed destinations as occupied ground.
    let mut strides: Vec<usize> = SpanOperands::widths(dims)
        .into_iter()
        .map(|cols| slot_stride(cap * cols * f32_bytes))
        .collect();
    strides.sort_unstable();
    strides.dedup();
    let mut planned = 0usize;
    let mut moved = 0usize;
    for bytes in strides {
        let moves = plan_slot_moves(device, SlotTenant::RewindStash, bytes, max_moves)?;
        planned += moves.len();
        let mut by_src: HashMap<u64, ArenaSlot> =
            moves.into_iter().map(|m| (m.src, m.dst)).collect();
        moved += stash.relocate(&mut by_src)?;
        // Destinations whose source this stash does not hold go back.
        drop(by_src);
    }
    Ok((planned, moved))
}

/// `n` layers of driver-memory operands — for a device with no reservation to carve
/// from (a CPU device, or a unit test), and for an empty cohort, which needs no
/// memory at all.
fn zeroed_layers(
    dims: &DeltaNetDims,
    cap: usize,
    dev: &Device,
    n: usize,
) -> Result<Vec<SpanOperands>> {
    (0..n)
        .map(|_| SpanOperands::zeros(dims, cap, dev))
        .collect()
}

/// `n` layers of operands in rewind-stash arena slots: one claim per operand width
/// covering every layer, so the arena window opens at most four times for the whole
/// stash rather than once per buffer.
#[cfg(feature = "cuda")]
fn stash_in_slots(
    dims: &DeltaNetDims,
    cap: usize,
    dev: &Device,
    n: usize,
) -> Result<Vec<SpanOperands>> {
    let f32_bytes = DType::F32.size_in_bytes();
    let mut per_width = SpanOperands::widths(dims)
        .into_iter()
        .map(|cols| {
            claim_arena_slots(dev, SlotTenant::RewindStash, cap * cols * f32_bytes, n)
                .map(|slots| slots.into_iter().map(Arc::new))
        })
        .collect::<Result<Vec<_>>>()?;
    (0..n)
        .map(|_| {
            let slots =
                [0, 1, 2, 3].map(|w| per_width[w].next().expect("n slots claimed per width"));
            SpanOperands::in_slots(dims, cap, dev, slots)
        })
        .collect()
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
        // [`SpanOperands::in_slots`] for the measurement that made this
        // necessary.
        #[cfg(feature = "cuda")]
        let layers = match dev {
            Device::Cuda(_) if cap > 0 && n > 0 => stash_in_slots(dims, cap, dev, n)?,
            _ => zeroed_layers(dims, cap, dev, n)?,
        };
        #[cfg(not(feature = "cuda"))]
        let layers = zeroed_layers(dims, cap, dev, n)?;
        Ok(Self {
            layers,
            spans: Vec::new(),
            cohort: 0,
            filled: vec![false; n],
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
        self.cohort = blocks.len();
        Ok(())
    }

    /// Spans the last [`Self::begin`] laid out, however many have since been
    /// removed — the span count the verify forward priced the replay from.
    pub fn cohort_spans(&self) -> usize {
        self.cohort
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

    /// Move every operand buffer whose slot is the source of a planned move onto
    /// that move's destination; answers how many moved.
    ///
    /// **Between steps, and only with no span outstanding.** A replay resolves the
    /// buffers' addresses when it runs, so moving them between steps is invisible —
    /// but a stash that still names spans is one a rewind may consume at any
    /// moment, and a caller must not hand it to a pass. [`Self::is_unused`] is that
    /// check.
    ///
    /// The stash's strides follow the verify cap rather than the model's geometry,
    /// so a cohort of a new width opens new pools. That is why this exists at all:
    /// the arenas of the caps that came before do not vanish, they empty, and their
    /// live remnants sit wherever the previous cohorts left them.
    #[cfg(feature = "cuda")]
    pub fn relocate(&mut self, moves: &mut HashMap<u64, ArenaSlot>) -> Result<usize> {
        let mut moved = 0usize;
        for ops in &mut self.layers {
            moved += ops.relocate(moves)?;
        }
        Ok(moved)
    }

    /// Whether any sequence still names a span in this stash.
    ///
    /// **The buffers outlive a span deliberately and must not outlive every
    /// span.** Keeping them across steps is the point — they are reallocated
    /// only when a wider cohort arrives — but once no sequence names one, the
    /// stash is holding rewind-stash arena slots on behalf of nobody, and it holds
    /// them for the life of the process. It is not KV, so every KV-side
    /// diagnostic reports the pool as healthy; only the tenant's own arena count
    /// shows the ground it keeps.
    pub fn is_unused(&self) -> bool {
        self.spans.is_empty()
    }
}

/// Per recurrent layer, in sweep order: the four small constants the mixer
/// needs, and the transformer-layer index they belong to.
///
/// The caller resolves these, because *where* they come from is the one
/// model-specific thing in a replay. A streamed checkpoint reads them from the
/// layer's residue rather than the layer — the replay runs at accept time, well
/// after the sweep that captured the stash, so the image may long since have
/// been evicted and `ensure`ing it would pull ~240 MB over PCIe to read four
/// constants that never left VRAM. A resident stack hands over the layer's own.
pub struct ReplayLayer<'a> {
    pub layer_index: usize,
    pub consts: DeltaNetConstants<'a>,
}

/// Advance each rewinding sequence's recurrent state to its accepted prefix,
/// from the state the block was entered with — on the device, the whole cohort
/// across a stack of recurrent layers in one launch triple
/// ([`replay_stacked`]), through the same span-table kernels the verify wave
/// itself ran.
///
/// Call **once per step**, immediately after the verify wave committed and
/// before any other wave touches these sequences: the entering states live in
/// each store's non-live half only until the next wave writes there.
///
/// A job whose `kept == span.len` is a full accept and is skipped without
/// touching anything — its live state already covers exactly those tokens.
///
/// Model-agnostic: everything specific to a checkpoint is resolved by the
/// caller into `layers` — see [`ReplayLayer`]. Both the hybrid and `qwen4exp`
/// run this, because the recurrence they rewind is the same one.
pub fn replay_accepted_prefixes(
    layers: &[ReplayLayer<'_>],
    dims: &DeltaNetDims,
    eps: f64,
    device: &Device,
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
    if layers.len() != layer_indices.len() {
        candle::bail!(
            "verify replay: {} layers supplied against {} recurrent layers — the caller's \
             sweep order and the store's disagree",
            layers.len(),
            layer_indices.len()
        );
    }
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
    for (ord, &li) in layer_indices.iter().enumerate() {
        if layers[ord].layer_index != li {
            candle::bail!(
                "verify replay: layer {} supplied where the store's ordinal {ord} is layer \
                 {li} — a replay against the wrong layer's constants advances the state \
                 silently and wrongly",
                layers[ord].layer_index
            );
        }
    }
    #[cfg(feature = "cuda")]
    if device.is_cuda() && dims.head_dim == DELTA_NET_PREFILL_DIM {
        return replay_stacked(layers, dims, eps, device, stash, &mut short, &layer_indices);
    }

    // **A generation for the replay, because the stash has no provenance to
    // lend.**
    //
    // `SpanOperands` lives outside any forward — the sequence owns it across
    // waves, which is the whole point of a rewind stash — so its tensors name no
    // wave. Every intermediate the mixer builds with `empty_beside` a stash
    // operand would land on the pool: `conved`, then `u`/`w`/`kq`/`g_cs`, then
    // everything downstream, per DeltaNet layer, per rewinding sequence, on
    // every accept.
    //
    // Measured with `--features forbidden_allocations` on the 27B: **20.0 GB** of
    // driver allocation at 20 contexts against 921 MB at one, on a card with
    // ~258 MiB outside the reservation. It surfaced as
    // `CUDA_ERROR_OUT_OF_MEMORY` on an unrelated event record, because by then
    // the device was simply full — and no region-pool diagnostic showed distress,
    // since none of it went through the pool.
    //
    // Opening a generation is not sufficient on its own — `empty_beside` relays
    // provenance rather than creating it, so the *root* must be leased. The root
    // is the conv's output, carved here; the kernels read the stash in place,
    // and the span table and the scan's buffers follow `conved` onto the wave.
    //
    // **The generation is per layer, not per replay.** The span is sized for one
    // layer's attention phase; holding one guard across the sweep accumulates
    // every layer's carves in it and exhausts it — measured, at layer 48 of the
    // first config: *"transient span exhausted — 491520 B at offset 23240704
    // exceeds the 23638784 B budget"*. Dropping the guard each iteration returns
    // the mixer's intermediates before the next layer asks, which is the same
    // lifetime a forward gives its phases.
    for (ord, &li) in layer_indices.iter().enumerate() {
        let entry = &layers[ord];
        #[cfg(feature = "cuda")]
        let wave: Option<WaveGeneration> = match device {
            Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?),
            _ => None,
        };
        let p = stash.layers[ord].all_rows();
        let stash_rows = p.qkv.dim(0)?;
        // Uninitialised: the conv writes every row the scan reads (invariant 6).
        #[cfg(feature = "cuda")]
        let conved = wave_empty(
            (stash_rows, dims.conv_dim()),
            DType::F32,
            device,
            wave.as_ref(),
        )?;
        #[cfg(not(feature = "cuda"))]
        let conved = Tensor::empty((stash_rows, dims.conv_dim()), DType::F32, device)?;
        let c = &entry.consts;
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
        delta_net_advance_spans(&p, c, dims, &mut seqs, eps, &conved)?;
    }
    Ok(())
}

/// [`replay_accepted_prefixes`] on the device: the recurrent layers in stacks
/// of [`DeltaNetWidths::replay_stack`], one launch triple a stack.
///
/// **The stack is sized from the stash's whole cohort, not from the spans that
/// rewind.** The verify forward priced the replay's carves from the stash's
/// capacity and its span count — it ran before anyone knew which blocks would
/// be cut short — and fewer spans make a layer cheaper, so sizing from the
/// short ones could stack more layers than the forward reserved room for.
///
/// One generation per stack, for the reason the per-layer path holds one per
/// layer: a stack's carves are what the forward priced for one replay launch,
/// and the guard's drop returns them before the next stack asks.
#[cfg(feature = "cuda")]
fn replay_stacked(
    layers: &[ReplayLayer<'_>],
    dims: &DeltaNetDims,
    eps: f64,
    device: &Device,
    stash: &VerifyStash,
    short: &mut [&mut (StashSpan, usize, &mut RecurrentStateStore)],
    layer_indices: &[usize],
) -> Result<()> {
    let Device::Cuda(cuda) = device else {
        candle::bail!("qwen35 verify replay: the stacked replay runs on a CUDA device");
    };
    let rows = stash.capacity()?;
    let widths = DeltaNetWidths {
        conv_dim: dims.conv_dim(),
        value_dim: dims.value_dim(),
        n_v_heads: dims.n_v_heads,
        layers: layer_indices.len(),
    };
    let cohort = stash.cohort_spans();
    let per = widths.replay_stack(rows, cohort);
    if per == 0 {
        candle::bail!(
            "qwen35 verify replay: a {rows}-row stash over {cohort} spans stacks no layer"
        );
    }
    let spans: Vec<ReplaySpan> = short
        .iter()
        .map(|job| ReplaySpan {
            start: job.0.row,
            len: job.1,
        })
        .collect();
    let projections: Vec<DeltaNetProjections<'static>> =
        stash.layers.iter().map(|l| l.all_rows()).collect();
    for first in (0..layer_indices.len()).step_by(per) {
        let ords = first..(first + per).min(layer_indices.len());
        let mut stacked: Vec<StackedLayer<'_, '_>> = Vec::with_capacity(ords.len());
        for ord in ords.clone() {
            let li = layer_indices[ord];
            // Each store's two halves of this layer, read as addresses: the
            // half the block entered with, and the live one the shorter advance
            // replaces. The borrow ends with the read; the buffers stay where
            // the stores hold them for the launch below.
            let states = short
                .iter_mut()
                .map(|job| {
                    let (entering, out) = job.2.layer_state_rewind(li)?;
                    ReplayStates::of(entering, &out, dims)
                })
                .collect::<Result<Vec<_>>>()?;
            stacked.push(StackedLayer {
                p: &projections[ord],
                c: &layers[ord].consts,
                states,
            });
        }
        let wave = begin_wave(&cuda.cuda_stream(), LayerPhase::Attention)?;
        // Uninitialised: the conv writes every row the scan reads (invariant 6).
        let conved = wave_empty(
            (ords.len() * rows, dims.conv_dim()),
            DType::F32,
            device,
            Some(&wave),
        )?;
        delta_net_replay_stack(&stacked, &spans, dims, eps as f32, &conved)?;
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

    /// **The four operand buffers must not alias each other.** Each is its own
    /// slot, so a wrong stride or a shared base would have one capture silently
    /// overwrite another's rows — and the failure is invisible: every shape still
    /// checks out, the replay just mixes two operands together and the rewound
    /// state comes back subtly wrong. Written into each in turn, the other three
    /// must be bit-unchanged.
    ///
    /// Also pins that the buffers arrive **zeroed**: a replay hands the mixer the
    /// whole `cap`-row buffer while only the captured span was filled, so the rows
    /// above the span are read before they are written.
    #[test]
    fn the_four_operand_buffers_are_distinct_and_zeroed() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let dims = DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 3,
        };
        let cap = 6usize;
        let stash = VerifyStash::new(&[LayerKind::DeltaNet], &dims, cap, &device).unwrap();
        let ops = &stash.layers[0];
        let widths = SpanOperands::widths(&dims);

        let read = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, t) in [&ops.qkv, &ops.z, &ops.beta_lin, &ops.alpha_lin]
            .into_iter()
            .enumerate()
        {
            assert_eq!(t.dims2().unwrap(), (cap, widths[i]), "operand {i} shape");
            assert!(read(t).iter().all(|&v| v == 0.0), "operand {i} not zeroed");
        }

        // Stamp each buffer with a distinct value and check the others are intact.
        let all = [&ops.qkv, &ops.z, &ops.beta_lin, &ops.alpha_lin];
        for (i, target) in all.iter().enumerate() {
            let mark = (i + 1) as f32 * 11.0;
            let stamp = Tensor::full(mark, target.dims2().unwrap(), &device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            target.slice_set(&stamp, 0, 0).unwrap();
            device.synchronize().unwrap();
            for (j, other) in all.iter().enumerate() {
                if j == i {
                    assert!(read(other).iter().all(|&v| v == mark), "operand {i} write");
                } else if j > i {
                    assert!(
                        read(other).iter().all(|&v| v == 0.0),
                        "writing operand {i} touched operand {j} — the slots alias",
                    );
                }
            }
        }
    }

    /// **A rewind consumes its spans before it replays them**, so the cohort the
    /// replay is sized from must outlive [`VerifyStash::remove`] — sizing from
    /// the spans left read zero and refused every hybrid rewind.
    #[test]
    fn the_cohort_survives_the_rewind_removing_its_spans() {
        let dims = DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 3,
        };
        let mut stash = VerifyStash::new(&[LayerKind::DeltaNet], &dims, 8, &Device::Cpu).unwrap();
        assert_eq!(stash.cohort_spans(), 0);
        stash.begin(&[(7, 3), (9, 2), (4, 1)]).unwrap();
        assert_eq!(stash.cohort_spans(), 3);
        for seq in [7, 9, 4] {
            stash.remove(seq);
        }
        assert!(stash.is_unused());
        assert_eq!(stash.cohort_spans(), 3);
        stash.begin(&[(5, 4)]).unwrap();
        assert_eq!(stash.cohort_spans(), 1);
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
