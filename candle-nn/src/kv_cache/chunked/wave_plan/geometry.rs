//! Describes the model shapes every buffer's size is derived from, with no
//! width in them.
//!
//! The geometry is width-free on purpose, because width is what admission is
//! deciding, and an assumed width baked in would make the plan answer a
//! question it is supposed to be asked. Each field states one fact about the
//! model or the session that the plan depends on, and the derived helpers give
//! the column counts and dtype rules. Facts fixed when the session is created,
//! such as whether norms emit packed q8a128, are stated to the plan rather than
//! priced both ways, because an upper bound that nothing can spend is ground the
//! weight side gave up for nothing.

use super::*;

/// The model shapes a layer's buffers are derived from.
///
/// Deliberately **width-free**: a wave's row count is an argument to the sizing
/// functions, not a field here. Width is what admission is deciding, so baking
/// an assumed width into the geometry would make the plan answer a question it
/// is supposed to be asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelGeometry {
    /// Model hidden size.
    pub hidden: usize,
    /// Tokens the LM head scores over. The head's logits are the whole of the
    /// forward phase's cost that scales with anything, and they are one row per
    /// sequence — see [`WaveBuffer::HeadLogits`].
    pub vocab: usize,
    /// FFN intermediate size. On a MoE model this is the *per-expert*
    /// intermediate, not the dense equivalent.
    pub intermediate: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// Experts each token routes to. `1` for a dense model, which collapses the
    /// MoE terms to the dense FFN shapes rather than needing a second branch.
    pub experts_per_tok: usize,
    /// Experts the router scores over — the width of its logits, which is a
    /// per-token buffer the FFN phase carries. `1` on a dense model, where the
    /// term degenerates to a column the router never allocates and costs two
    /// bytes a row.
    pub n_experts: usize,
    /// The mixer widths of a **DeltaNet** layer, on a hybrid stack where only
    /// some layers attend. `None` where every layer attends.
    ///
    /// A DeltaNet layer's mixer carves from the *attention* arena — it is the
    /// other thing a layer can open that generation for — so its widths belong
    /// here and not in the FFN phase. Both kinds run an ordinary FFN, so the
    /// FFN phase is uniform and takes no term from this.
    pub delta_net: Option<DeltaNetWidths>,
    /// The MoE layer's always-active shared expert, where it has one. `None`
    /// on a MoE without one (Qwen3-MoE) and on a dense stack.
    pub shared_expert: Option<SharedExpertWidths>,
    /// The compute dtype activations are carried in.
    pub act_dtype: DType,
    /// What the int8 tensor-core kernels emit before the cast back to
    /// `act_dtype`. Both are live at once, so both are planned.
    pub accum_dtype: DType,
    /// Whether the layer norms emit **q8a128** rather than the compute dtype.
    ///
    /// An int8 session's RMSNorm fuses its quantize into the norm's epilogue,
    /// so the buffer the projections consume is packed — about 1.8× smaller
    /// than the same rows in BF16. A float session's norm emits `act_dtype`.
    ///
    /// This is fixed when the session is created, so it is a fact the plan can
    /// be told rather than a case it has to bound. It used to be bounded: both
    /// norms were priced dense "because the two encodings are alternatives and
    /// pricing the larger keeps the plan an upper bound", which cost 1.8 MiB a
    /// phase on the 0.8B — the entire remaining gap in the FFN span once its
    /// chain was right. An upper bound that nobody can spend is ground the
    /// weight side conceded for nothing.
    pub packed_norm: bool,
    /// The same question asked of the **head**, whose answer can differ.
    ///
    /// The final norm is handed `output_proj().int8mode()` — the head weight's
    /// own mode — not a layer's, and a weight that could not be KO-repacked
    /// stays on the dequant path while every layer around it runs int8.
    /// Measured: Qwen2's layers are packed and its head is not, so the head
    /// carves a float norm and its F32 working copy while the logits leave the
    /// span. Priced from `packed_norm`, that phase was 98% slack — 17.1 MiB of
    /// a 17.4 MiB span, all of it logits that were never there.
    pub packed_head: bool,
    /// The Q projection emits `2 × head_dim` per head — interleaved
    /// `[q | gate]` — and the gate travels the whole attention block: split
    /// out contiguously, sigmoided, and multiplied into the context. Widens
    /// `qkv_cols` and charges the three gate-side buffers. Gated lineages
    /// (Qwen3.5/3.8) set it; classic attention leaves it false and pays
    /// nothing.
    pub gated_qkv: bool,
    /// Whether Q, K and V come out of **one** segmented projection and are
    /// narrowed out of it, rather than from three separate matmuls.
    ///
    /// A fused projection writes one `qkv_cols`-wide row, so each of the three
    /// is a strided view and reaching it costs a copy — `QSplit`, `KSplit`,
    /// `VContiguous`. Three separate `forward_dynamic` calls write three
    /// contiguous buffers and none of those copies happens; the widths are
    /// identical either way, which is why [`WaveBuffer::QkvProjection`] prices
    /// both with one charge and only the splits turn on this.
    ///
    /// **Usually a property of the session, not the model.** Every model on the
    /// generic path forks on the operand it is handed: `DynamicActs::Int8` takes
    /// `QMatMul::qkv_segmented` (one launch, three narrows), `DynamicActs::Float`
    /// takes three separate `forward_dynamic` calls and narrows nothing. So it
    /// is derived from the int8 mode there, exactly as [`Self::packed_norm`] is.
    /// The Qwen3.5 lineage is the exception and is `false` in both modes — its Q
    /// weight is the interleaved `[q | gate]` and does not pack with K and V.
    pub fused_qkv: bool,
    /// Whether Q, K and V each get a **bias** added after the projection
    /// (Qwen2).
    ///
    /// The add is an allocation at every width, and it reads the narrowed view
    /// as it stands — so on a fused projection it *replaces* the split copy
    /// rather than following it. Qwen2 used to be priced through the split
    /// copies, whose sizes happen to match; that held until a one-row wave,
    /// where the narrows are free and the adds are not, and the span ran out
    /// 1,792 B into the Q bias.
    pub qkv_bias: bool,
    /// Whether the paged decode kernel emits its context **as q8a1024 on the
    /// span** — an int8 session at a head dim the fused q8 combine serves
    /// (`prefill_utils::paged_decode_q8_head_dim`, the one predicate the
    /// dispatch itself goes through; it cannot be called from this crate, so
    /// the model states the answer here).
    ///
    /// Elsewhere the decode path takes the FP context and the chain carves one
    /// `rows × hidden` buffer after the projection rather than a q8 context
    /// *and* one: measured on Qwen2 (head dim 64) at 60 decode rows, 445,184 B
    /// against the 506,368 B the plan charged with the context in it — exactly
    /// the context and its alignment pad.
    pub decode_q8_context: bool,
    /// Whether the model applies a **per-head RMSNorm to Q and K** at all.
    ///
    /// Qwen3 and later carry one; Llama and Qwen2 do not, and charging them for
    /// `QNormOut`/`KNormOut` — plus the four reshape copies below — priced six
    /// buffers those stacks never allocate, two of them `attn_cols` wide.
    ///
    /// Separate from [`Self::head_norm_reshapes`] because they are different
    /// questions: this is whether the norm exists, that is whether reaching it
    /// costs copies.
    pub head_qk_norm: bool,
    /// Whether the per-head Q/K RMSNorms need a flatten in and a transpose out.
    /// Meaningless, and ignored, when [`Self::head_qk_norm`] is false.
    ///
    /// The norm reduces over the head dim, so it wants `[.., heads · seq, dim]`.
    /// A wave carried as `[batch, seq, heads · dim]` cannot reach that by
    /// reshaping — the transpose is not contiguous — so it copies in and copies
    /// back: `QNormIn`, `QHeadsPacked`, `KNormIn`, `KHeadsPacked`, four buffers
    /// and the two widest of them `attn_cols` wide. A wave already packed as
    /// `[total_rows, cols]` reshapes for free and carves none of them.
    ///
    /// Set by the shape the forward hands the layer, not by the model's
    /// weights: the Qwen3.5 wave flattens to `[rows, hidden]` before the layer
    /// sweep and so leaves this false.
    pub head_norm_reshapes: bool,
    /// Only part of the head width rotates, and the paged kernels only know
    /// full-width RoPE, so Q and K are re-ordered through a gather
    /// (`RotaryLayout::permute_last_dim_live`) — one `attn_cols`-wide and one
    /// `kv_cols`-wide copy per layer. Full-width-rotary models leave it false.
    pub partial_rotary: bool,
    /// The Gated Residual's pre-mix widths, on a multi-stream stack — `None`
    /// on every stack with ordinary layer norms. See [`HyperWidths`].
    ///
    /// Where it is set, the pre-mix **is** each phase's norm: it carves the
    /// block input the mixer and the FFN read, so the float norms
    /// ([`WaveBuffer::AttnNorm`] on a float session, [`WaveBuffer::DeltaNetNorm`],
    /// [`WaveBuffer::FfnNorm`]) carve nothing of their own.
    pub hyper: Option<HyperWidths>,
    /// Whether a **single-stream** stack carries a NextN draft head, whose pass
    /// runs over every row of a wave that reaches the head — the `Mtp*`
    /// buffers. A multi-stream stack states its head in
    /// [`HyperWidths::draft_head`] instead.
    pub mtp_head: bool,
}

impl ModelGeometry {
    /// Rows the expert GEMMs see: every token replicated to each expert it
    /// routed to. This is what makes a MoE layer's FFN phase several times a
    /// dense one's, and why admission prices MoE waves differently.
    pub const fn expert_rows(&self, rows: usize) -> usize {
        rows * self.experts_per_tok
    }

    /// Columns produced by the fused QKV projection. A gated lineage's Q
    /// projection emits `[q | gate]` — twice the query width.
    pub const fn qkv_cols(&self) -> usize {
        let q_cols = if self.gated_qkv {
            2 * self.n_head
        } else {
            self.n_head
        };
        (q_cols + 2 * self.n_kv_head) * self.head_dim
    }

    /// Columns of the attention output, before `o_proj`.
    pub const fn attn_cols(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Columns of one of K or V, as the QKV narrow produces them.
    pub const fn kv_cols(&self) -> usize {
        self.n_kv_head * self.head_dim
    }

    /// The width the FFN's intermediates are carried in.
    ///
    /// `forward_layer_batched_mixed` widens an F16 activation to BF16 for the
    /// SwiGLU, whose range can exceed F16's, and leaves every other dtype
    /// alone. Derived rather than carried, because it is one rule stated in one
    /// place in the forward and a second copy of it would drift.
    ///
    /// The width is the same two bytes either way, so this changes no buffer's
    /// *size* — what it decides is whether two casts are allocations or no-ops:
    /// [`WaveBuffer::FfnNormOperand`] on the way in, and
    /// [`WaveBuffer::MoeResultCast`] on the way back out, where `to_dtype_mut`
    /// returns early only when the dtypes already agree.
    pub fn work_dtype(&self) -> DType {
        ffn_work_dtype(self.act_dtype)
    }

    /// Whether a layer's FFN dispatches to the expert pipeline or to a single
    /// dense MLP — which of [`Chain::Ffn`] and [`Chain::DenseFfn`] runs.
    ///
    /// Read from the router's width rather than carried as its own flag: a
    /// model with one expert has no router to score and no fan-out to apply, so
    /// `n_experts > 1` is the same question asked of the geometry that is
    /// already here.
    pub const fn is_moe(&self) -> bool {
        self.n_experts > 1
    }
}
