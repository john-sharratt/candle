//! Holds the widths of the model features that are not plain attention or FFN:
//! DeltaNet, the shared expert and the Gated Residual.
//!
//! DeltaNet's mixer stays F32 throughout, because its recurrent state sums over
//! every token of a sequence, where half precision drifts without bound with
//! context length; only the final `w_out` result narrows to the residual's
//! dtype. A shared expert lists every step of its always-active chain, since an
//! uncharged carve there would leave the routed down projection without room.
//! The Gated Residual's widths describe the parallel residual streams and their
//! pre-mix, which runs in F32 whatever the session's dtype.

use super::*;

/// A DeltaNet mixer's widths, as [`ModelGeometry::delta_net`] carries them.
///
/// Everything the mixer holds is F32 and stays F32: `S` is a running sum over
/// every token of a sequence, the one value in the stack with no bound on how
/// many additions it accumulates, and half precision drifts without bound in
/// context length — the opposite of what the O(1)-error design is for. So the
/// four projections ask the KO kernel to store F32 out of the accumulator it
/// already has, and the output projection reads that F32 directly. Only the
/// final `w_out` result narrows, to whatever the residual stream carries.
/// The widths price the live mixer's carves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaNetWidths {
    /// `2 · key_dim + value_dim` — the fused `[Q|K|V]` projection's width.
    pub conv_dim: usize,
    /// `head_dim × n_v_heads` — `z`'s width, and the mixer's output.
    pub value_dim: usize,
    /// V heads. `beta` and `alpha` are one F32 scalar per head per row.
    pub n_v_heads: usize,
    /// DeltaNet layers in the stack. The forward's decode table carries three
    /// addresses per decode row for each of them — the conv tail, the state and
    /// the timeline.
    pub layers: usize,
}

/// A MoE layer's always-active **shared expert**, as
/// [`ModelGeometry::shared_expert`] carries it.
///
/// An ordinary SwiGLU every token goes through, scaled by a per-token
/// `sigmoid(w_gate · x)` and summed with the routed combine — Qwen3.5/3.6's
/// MoE. Every step of it lands on the span: the fused gate/up projection, the
/// SiLU, the product, the down projection's q8a128 operand and its result, the
/// gate's tile-wide projection, its sigmoid, the gated product, and the final
/// sum. Measured on Qwen3.5-35B-A3B at 2,100 rows: those nine carves.
///
/// **An uncharged carve here is not slack somewhere else.** The FFN generation
/// is sized to exactly this chain, and the shared half carves before the routed
/// one, so a shared buffer the plan omits leaves the routed down projection —
/// the chain's last large carve — without room, and the wave fails on it
/// (`WaveCarve::Exhausted`) while the tier reserves bytes nothing ever carves:
/// both symptoms of one missing declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedExpertWidths {
    /// The shared SwiGLU's intermediate width.
    pub intermediate: usize,
    /// Columns the gate projection emits: its one real output, padded to a full
    /// KO tile so the int8 kernel can run it.
    pub gate_cols: usize,
}

/// The Gated Residual's widths, as [`ModelGeometry::hyper`] carries them.
///
/// A stack that carries its residual as several parallel streams and has no
/// layer norms (Qwen3.8-Flash-Next): each layer phase opens with a **pre-mix**
/// that collapses the wide residual to the block input — a grouped norm over
/// every stream, a low-rank read gate, and the gated mean — and that pre-mix is
/// the phase's seed, so everything it carves lands on the span. It runs in F32
/// whatever the session's dtype: the residual it reads is F32 the whole way
/// round the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperWidths {
    /// Parallel residual streams.
    pub streams: usize,
    /// The read gate's bottleneck rank.
    pub low_rank: usize,
    /// Whether a draft head runs over the wave's rows after the trunk: one
    /// attention block whose input is assembled from the shifted wide residual
    /// and the rows' embeddings ([`Chain::HeadPass`]).
    pub draft_head: bool,
    /// Whether the stack's per-layer n-gram embedding block runs in the
    /// forward, ahead of its layer's mixer ([`Chain::Ple`]).
    pub ple: bool,
    /// What an attention mixer carves its projections and per-head
    /// intermediates in: the wave's own dtype. Only the residual is F32 the
    /// whole way round, so pricing the mixer in
    /// [`ModelGeometry::act_dtype`] — F32 on this stack — doubles every Q, gate
    /// and context buffer: measured on Qwen3.8-Flash-Next at BF16 ×4, an
    /// attention phase priced at 1,122,580,736 B whose widest generation carved
    /// 983,476,480 B. See [`WaveBuffer::is_mixer_carve`].
    pub mixer_dtype: DType,
}
