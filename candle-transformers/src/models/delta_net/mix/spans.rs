//! The geometry of the sequences packed into one wave buffer, and the per-layer
//! pointer tables that let one launch serve all of them.
//!
//! Each sequence is a run of rows in the buffer with the state it carries, and
//! the runs must tile the buffer in order. A short span writes a timeline that is
//! folded into its state later, so a partial accept can keep a prefix of it. A
//! long span writes its advanced state into scratch that the wave commits. The
//! pointer tables are uploaded once per forward rather than once per layer, since
//! a per-layer upload would stall the launch pipeline. They live on the wave's
//! transient tier, so they are released when that generation resets.

use super::*;

/// One sequence's run of rows in a packed wave buffer, with the state it
/// carries.
///
/// The mixer is **row-wise except in two places**: the causal conv carries a
/// tail and the delta rule carries `S`, and both are per sequence. Everything
/// else — the projections, the gates, the norms, the GQA broadcast, the output
/// gate — computes each row from that row alone. Passing the whole wave and
/// naming the sequence boundaries is what lets the row-wise majority run once
/// instead of once per sequence.
pub struct DeltaNetSeq<'a> {
    /// First row of this sequence in the packed buffer.
    pub start: usize,
    /// How many rows it contributes.
    pub len: usize,
    /// The recurrent state this sequence enters the wave with. Read, never
    /// written: a short span's contribution lands in its timeline and a long
    /// span's advanced state in [`SpanWrite::State`].
    pub state: &'a mut DeltaNetState,
    /// Where the span's work lands.
    pub write: SpanWrite,
    /// A capture of the span's scan inputs, or a replay that writes its
    /// advanced state from a record's rows (`span_arm`). Long spans on the
    /// fused path only.
    pub hook: SpanHook,
}

/// Where a span's work lands — chosen by its length, never by its purpose
/// (`docs/deltanet_timeline.md` §7). A clone is another handle on the same
/// buffers.
#[derive(Debug, Clone)]
pub enum SpanWrite {
    /// A span of at most [`TIMELINE_ROWS`] rows: its timeline, a
    /// [`TimelineLayout`] buffer the span writes one row per token into. The
    /// state and the conv tail are read and left as they are; the fold applies
    /// the rows the step keeps.
    Timeline(Tensor),
    /// A longer span: the buffers its advanced state and conv tail are written
    /// to — the store's prefill scratch, which `commit_wave` exchanges with
    /// the live state.
    State(DeltaNetOut),
}

impl SpanWrite {
    pub fn timeline(&self) -> Option<&Tensor> {
        match self {
            Self::Timeline(t) => Some(t),
            Self::State(_) => None,
        }
    }

    pub fn state(&self) -> Option<&DeltaNetOut> {
        match self {
            Self::Timeline(_) => None,
            Self::State(o) => Some(o),
        }
    }
}

/// Check that every span's write fits its length: a timeline for a span of at
/// most [`TIMELINE_ROWS`] rows, a state for a longer one.
pub(super) fn check_span_writes(seqs: &[DeltaNetSeq<'_>]) -> Result<()> {
    for (i, s) in seqs.iter().enumerate() {
        let short = s.len <= TIMELINE_ROWS;
        if short != s.write.timeline().is_some() {
            candle::bail!(
                "delta_net_mix_spans: span {i} of {} rows was handed a {} — a span of at \
                 most {TIMELINE_ROWS} rows writes a timeline, a longer one a state",
                s.len,
                if short { "state" } else { "timeline" }
            );
        }
    }
    Ok(())
}

/// The write for a single span over **one** carried state, outside any store —
/// the single-span entry points and the reference tests, which have no wave to
/// roll back. A short span writes a timeline of its own, which
/// [`finish_solo`] folds whole; a longer one writes `s` in place and its conv
/// tail to a scratch buffer [`finish_solo`] copies back.
pub fn solo_write(state: &DeltaNetState, len: usize, dims: &DeltaNetDims) -> Result<SpanWrite> {
    if len <= TIMELINE_ROWS {
        Ok(SpanWrite::Timeline(
            TimelineLayout::of(dims).alloc(state.s.device())?,
        ))
    } else {
        Ok(SpanWrite::State(state.solo_out()?))
    }
}

/// Land a [`solo_write`] span's work in its state: fold every row of a
/// timeline, or take the advanced conv tail back from its scratch buffer.
pub fn finish_solo(
    state: &mut DeltaNetState,
    write: &SpanWrite,
    len: usize,
    dims: &DeltaNetDims,
) -> Result<()> {
    match write {
        SpanWrite::Timeline(tl) => fold_one(state, tl, len, dims),
        SpanWrite::State(out) => state.absorb_solo(out),
    }
}

/// Fold the first `m` rows of `tl` into `state` — the fold kernel on a CUDA
/// device at the kernels' head width, the tensor-op fold elsewhere.
pub fn fold_one(
    state: &mut DeltaNetState,
    tl: &Tensor,
    m: usize,
    dims: &DeltaNetDims,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    if state.s.device().is_cuda() && super::super::cuda::fold_runs_fused(dims) {
        return super::super::cuda::delta_net_fold_states(&[(&*state, tl, m)], dims);
    }
    fold_by_ops(state, tl, m, dims)
}

/// Where one sequence's rows sit in the wave's packed activation buffer.
///
/// The buffer is flat `[1, total_tokens, hidden]` with each sequence a
/// contiguous run, so a recurrent mixer only needs the run's start and
/// length to take its own slice. [`DeltaNetSeq`] is this plus the state the
/// run advances; a `SeqSpan` is the pure geometry, used where the state is
/// looked up separately (the sweep, the decode pointer table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqSpan {
    /// The sequence's id, which is what keys its recurrent state.
    pub seq: usize,
    /// First row of this sequence in the packed buffer.
    pub start: usize,
    /// How many rows it contributes to this wave.
    pub len: usize,
}

/// Build the per-sequence spans of a packed wave buffer.
///
/// `q_lens` is in the same order the buffer was packed in, which is the
/// engine's `[decode | prefill | glue]` order — one row per decode sequence,
/// `q_len` rows per prefill or glue sequence.
pub fn seq_spans(seqs: &[usize], q_lens: &[usize]) -> Result<Vec<SeqSpan>> {
    if seqs.len() != q_lens.len() {
        candle::bail!(
            "seq_spans: {} sequences against {} lengths",
            seqs.len(),
            q_lens.len()
        );
    }
    let mut spans = Vec::with_capacity(seqs.len());
    let mut start = 0usize;
    for (&seq, &len) in seqs.iter().zip(q_lens) {
        spans.push(SeqSpan { seq, start, len });
        start += len;
    }
    Ok(spans)
}

/// One layer's slice of the decode pointer table: device arrays of each
/// decode sequence's conv-tail, state and timeline base pointers
/// (`ptrs [3, n_decode]` I64) plus each sequence's row in the wave tensors
/// (`rows [n_decode]` U32).
///
/// Decode states live in per-session allocations, so batching the decode
/// conv and recurrence into one launch each needs their addresses on the
/// device. The wave driver builds the whole forward's table in ONE host
/// upload before the layer sweep ([`super::cuda::build_wave_table`]) — never
/// per layer, where the upload's stream sync would serialise the launch
/// pipeline — and each layer receives its slice. Pointers are stable within the
/// forward — the store holds the buffers and their anchors keep the slots held
/// — but not across one: compaction moves slots between forwards, so a table
/// is never kept past the forward that built it. `'w` because a slice of
/// [`super::cuda::DeltaNetWaveTable`] borrows it, and that table lives on the
/// forward-phase span — so this must not outlive the generation whose reset
/// reclaims it. A per-layer table built standalone is `'static`, which this
/// still admits.
pub struct DeltaNetLayerTable<'w> {
    /// `[3, n_decode]` I64 device tensor (or a view of the wave table): conv
    /// tail, state, timeline.
    pub ptrs: LiveTensor<'w>,
    /// `[n_decode]` U32 wave rows.
    pub rows: LiveTensor<'w>,
}

/// The same table for the wave's **multi-row** spans, in one of two forms by
/// length: a short table (`[3, n]` — conv tail, state, timeline) for spans of at
/// most [`TIMELINE_ROWS`] rows, which the short-span kernel runs, and a long
/// table (`[4, n]` — conv tail in, conv tail out, state in, state out) for the
/// prefill spans the conv + scan pair runs.
///
/// A span differs from a decode row only in occupying `len` consecutive rows
/// instead of one, so it needs the same pointers plus that extent. With
/// them on the device, every span in the wave runs in one launch per kernel
/// rather than one launch per span (hot-path invariant 5) — which is what the
/// decode rows next door have always done.
///
/// The saving is launches, not arithmetic: each span still walks its own rows
/// in order, because `S` is a running sum within a sequence. What changes is
/// that four sequences no longer serialise behind each other to do it.
/// The table is **wave-scoped**, and the lifetime says so.
///
/// Its two uploads are rebuilt every wave (the pointers and the packing extents
/// are what change), so they belong on the wave's transient tier rather than in
/// driver memory the reservation does not cover. Placing them there means they
/// die when the generation's cursor rewinds — which is precisely what `'w`
/// encodes, and why the fields cannot be plain `Tensor`: `Tensor` is
/// `LiveTensor<'static>`, and a `'static` handle onto recycled arena memory is
/// the use-after-reset the lifetimes exist to prevent.
pub struct DeltaNetSpanTable<'w> {
    /// `[3, n]` I64 for a short table, `[4, n]` for a long one.
    pub ptrs: LiveTensor<'w>,
    /// `[2, n]` U32: first row in the packed wave buffer, then row count.
    pub spans: LiveTensor<'w>,
    pub n: usize,
    /// Rows in the longest span — the launch's grid extent over tokens.
    pub max_len: usize,
}

/// The spans must tile the packed buffer exactly, in order — see
/// [`delta_net_mix_spans`].
pub(super) fn check_span_tiling(seqs: &[DeltaNetSeq<'_>], t: usize) -> Result<()> {
    let mut cursor = 0usize;
    for (i, s) in seqs.iter().enumerate() {
        if s.start != cursor {
            candle::bail!(
                "delta_net_mix_spans: sequence {i} starts at {} but the previous \
                 one ended at {cursor} — spans must tile the packed buffer",
                s.start
            );
        }
        cursor += s.len;
    }
    if cursor != t {
        candle::bail!("delta_net_mix_spans: spans cover {cursor} rows but the buffer holds {t}");
    }
    Ok(())
}
