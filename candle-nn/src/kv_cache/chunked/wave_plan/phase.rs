//! Defines the generations a layer's buffers live in and how their spans are
//! laid out in the transient tier.
//!
//! A layer opens an attention generation and an FFN generation that never
//! overlap, so the two share one region of the tier, with the forward span
//! standing beside them because it holds what crosses every layer. Each span
//! starts on the same quantum the carves align to, so an entry that is not a
//! multiple of it would shift the forward span off the boundary and leave the
//! carves' alignment unsupported. `launch_plan` adds the launchers' scratch to
//! the forward span and aligns every entry, so admission prices the wave the
//! forward actually places.

use super::*;

/// Alignment every bump range is rounded up to.
///
/// What the tensor-core paths require of their operands. A phase of `n` buffers
/// can therefore consume up to `n` alignments beyond the sum of its sizes, and
/// [`WavePlan::wave_bytes`] charges for exactly that.
pub const BUMP_ALIGNMENT: usize = 256;

/// Which generation a buffer lives in.
///
/// Named `LayerPhase` rather than `WavePhase` because
/// `batched_model::WavePhase` already means something else — the result of a
/// layer *range*, not a scope within one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum LayerPhase {
    /// Attention norm through `o_proj`. Reset every layer.
    Attention,
    /// FFN norm through the MoE combine. Reset every layer.
    Ffn,
    /// The whole forward, rather than one layer of it.
    ///
    /// For the setup a forward builds once and reads from *every* layer — the
    /// ragged prefill metadata, the RoPE tables, the gathered position ids — and
    /// for the head that runs after the last layer. None of it fits the layer
    /// phases: a buffer carved from the attention span is reclaimed when layer
    /// 0's guard drops, and layer 1 would overwrite the tables it is still
    /// reading.
    ///
    /// Its width is [`WaveWidth::scored_rows`], not the wave's rows: the head
    /// produces logits for one row per sequence, not one per token. That is the
    /// whole reason this phase is separate rather than folded into the layer
    /// phases, which scale with tokens.
    Forward,
}

/// The bytes a tier holding spans of `per_phase` (attention, FFN, forward)
/// occupies: the two **layer** phases share one region — the larger of the
/// two — and the forward span stands beside it.
///
/// A layer's attention generation drops before its FFN generation opens, and
/// `begin_wave` refuses to open one while the other is live, so the two spans
/// are never written at once and can lie on the same ground. The forward span
/// cannot join them: it holds what crosses every layer — the residual, the
/// ragged-prefill tables, the head — and is live for the whole forward.
///
/// Laid side by side, the tier cost the sum of the three; on Qwen3.8-Flash-Next
/// at 3,072 rows that was ~1.2 GB of DeltaNet mixer and ~1.0 GB of FFN for
/// spans that are never live together.
pub fn tier_footprint(per_phase: [usize; 3]) -> usize {
    let [attention, ffn, forward] = per_phase;
    attention.max(ffn) + forward
}

/// Where each span of a tier holding `per_phase` starts, as an offset from the
/// tier's base, in the order of `per_phase`: the attention and FFN spans both
/// at the base, the forward span past the larger of them. The layout
/// [`tier_footprint`] prices.
pub fn phase_offsets(per_phase: [usize; 3]) -> [usize; 3] {
    let [attention, ffn, _] = per_phase;
    [0, 0, attention.max(ffn)]
}

/// The boundary every phase span's base must land on.
///
/// The same quantum ops carve at (`INHERIT_ALIGN` in candle-core), because a
/// span's base is laid out from the entries before it ([`phase_offsets`]): an
/// entry that is not a multiple of this shifts the forward span off the
/// boundary, and a carve's alignment is only as good as its span's base.
pub(crate) const WAVE_SPAN_ALIGN: usize = 256;

/// A phase plan with every entry rounded up to [`WAVE_SPAN_ALIGN`].
pub(crate) fn align_phase_plan(plan: [usize; 3]) -> [usize; 3] {
    plan.map(|bytes| bytes.div_ceil(WAVE_SPAN_ALIGN) * WAVE_SPAN_ALIGN)
}

/// The spans a forward lays out for buffers of `per_phase`: the forward-wide
/// span widened by `launch_scratch` — what the launchers carve inside their
/// launches (`candle::cuda_backend::scratch_reserve`) — and every entry
/// aligned. The one normalisation the tier's price ([`WavePlan::tier_bytes`])
/// and its placement (`plan_wave_transient`) both apply, so the wave admission
/// prices is the wave the forward places.
pub fn launch_plan(per_phase: [usize; 3], launch_scratch: usize) -> [usize; 3] {
    let [attention, ffn, forward] = per_phase;
    align_phase_plan([attention, ffn, forward + launch_scratch])
}
