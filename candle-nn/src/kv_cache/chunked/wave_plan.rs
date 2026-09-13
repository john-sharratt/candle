//! What a wave costs in the transient tier, and how wide a wave may therefore be.
//!
//! The wave domain's halves used to be a hand-tuned constant justified by a
//! measurement: 64 MiB against an observed 30.8 MiB peak on one model at one
//! batch size. That holds only while the half carries a single buffer. Once a
//! layer's whole working set comes from the transient tier, the cost scales
//! with the wave's width — and with `experts_per_tok` on a MoE model — so no
//! constant is right for both a decode wave and a wide prefill.
//!
//! # The budget decides the width, not the other way round
//!
//! The half size is a **policy input**: how much VRAM to lend the forward,
//! given that every megabyte here is a megabyte the KV side cannot hold.
//! Admission then asks this module how many rows fit in that budget and stops
//! there. Sizing the arena from an assumed worst-case width would be the same
//! arithmetic run backwards, and it would be a guess — a wave wider than the
//! guess exhausts the span and fails the forward, which is exactly the failure
//! the gate exists to prevent.
//!
//! Deriving the width from the budget makes the plan an upper bound *by
//! construction*: the scheduler cannot admit a wave it has not already priced.
//!
//! # One source of truth, and how it is held to that
//!
//! The failure mode this module is shaped to avoid is a plan that quietly
//! disagrees with what the code allocates. A buffer is named **once**, as a
//! [`WaveBuffer`] variant, and that variant answers two questions: which phase
//! it belongs to, and what shape it is at a given row count. The byte
//! arithmetic appears in exactly one function ([`BufferShape::bytes`]). Nothing
//! is kept in a parallel list — [`WaveBuffer::iter`] derives the inventory, so
//! the plan, the totals, the admission bound and the tests all pick up a new
//! variant with no edit.
//!
//! What the compiler cannot check is whether the list is *complete*, and that
//! is not a theoretical gap — it was wrong by a factor of 1.8 on the attention
//! phase and by 2 on the accumulate dtype, and every test stayed green. Under
//! operand provenance an op reading a wave-backed operand carves its output from
//! the same generation, so an undeclared buffer never reaches the driver and
//! never appears in a `candle::forbidden_alloc` report. It just costs the span.
//!
//! The check is therefore external and empirical:
//! [`super::wave_census`] itemises the layer that set each span's high-water
//! mark, in carve order, with the caller that asked for it. Every variant below
//! was read off that census on Qwen3-30B-A3B rather than inferred from the
//! source, and the totals in the tests are pinned against it. **A change to the
//! attention or FFN chain is a change to this list**, and a `wave-census`
//! build over the gate is how to find out what it should say.
//!
//! # A phase at a time, and the chains that can run in it
//!
//! A layer opens two generations — one spanning attention → `o_proj`, one
//! spanning the FFN — and each drops before the next opens, resetting its span.
//! So a span holds **one layer phase**, never a whole wave, and each phase gets
//! its own arena ([`super::bump_arena`]) sized from its own peak rather than
//! both from the larger. A third, the forward phase, holds what the head needs
//! after the last layer.
//!
//! Within a phase more than one [`Chain`] can run, and they combine two
//! different ways depending on *why* there is more than one:
//!
//! * **Two kinds of layer are a `max`.** A hybrid stack's layer has an
//!   attention mixer or a DeltaNet one, never both, and a generation is one
//!   layer's phase — so the span is sized by the larger. The same holds one
//!   phase down, where a layer's FFN dispatches to the expert pipeline or to a
//!   dense MLP. Summing those would price a wave for a layer that does not
//!   exist.
//! * **Two groups of one kind are a `sum`, each at its own width.**
//!   `forward_layer_batched_mixed` opens **one** attention generation and runs
//!   every group inside it, so a wave carrying a decode group and a prefill
//!   group holds both chains' buffers before either guard drops. Each is sized
//!   by the rows *its* group contributed, which is what [`WaveWidth`] carries
//!   and a single row count could not say.
//!
//! That second point used to be a bound rather than a measurement: handed one
//! total, the plan charged the decode chain at the whole wave's width and a
//! pure-prefill wave paid about 15% of its attention span for buffers it never
//! allocated. Passing the split closed it, and the plan now prices every phase
//! to the byte of what the census measures it carving.

use super::types::TARGET_ARENA_BYTES;
use candle::DType;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

/// Elements per q8a128 tile, and tiles per flat-grouped super-block.
///
/// The packing is eight 128-element tiles to a 1152-byte super-block
/// (`blocks.cuh`). Named rather than inlined because a reader checking this
/// arithmetic against the kernel should find both numbers in one place.
const Q8A128_TILE_ELEMS: usize = 128;
const Q8A128_TILES_PER_BLOCK: usize = 8;
const Q8A128_BLOCK_BYTES: usize = 1152;

/// Alignment every bump range is rounded up to.
///
/// What the tensor-core paths require of their operands. A phase of `n` buffers
/// can therefore consume up to `n` alignments beyond the sum of its sizes, and
/// [`WavePlan::wave_bytes`] charges for exactly that.
pub const BUMP_ALIGNMENT: usize = 256;

/// How a buffer's elements are laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Plain row-major elements of a candle dtype.
    Dense(DType),
    /// q8a128 flat-grouped super-blocks, as the int8 tensor-core path consumes.
    Q8a128,
}

/// A buffer's shape and encoding — everything needed to size it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferShape {
    pub rows: usize,
    pub cols: usize,
    pub encoding: Encoding,
}

impl BufferShape {
    /// The one place a shape becomes a byte count.
    ///
    /// `div_ceil` on both axes deliberately: a partial tile costs a whole tile
    /// and a partial group a whole super-block. Sizing is an upper bound, and an
    /// upper bound rounds up.
    pub fn bytes(&self) -> usize {
        match self.encoding {
            Encoding::Dense(dtype) => self.rows * self.cols * dtype.size_in_bytes(),
            Encoding::Q8a128 => {
                let tiles = self.rows * self.cols.div_ceil(Q8A128_TILE_ELEMS);
                tiles.div_ceil(Q8A128_TILES_PER_BLOCK) * Q8A128_BLOCK_BYTES
            }
        }
    }
}

/// How wide a wave is, in the three units its buffers actually scale with.
///
/// A single row count cannot price a wave, and the two places it fails are not
/// small:
///
/// * **A phase runs more than one chain.** `forward_layer_batched_mixed` opens
///   one attention generation and runs the decode group and the prefill group
///   inside it, so both chains' buffers are live at once — but each is sized by
///   *its own* group's rows. Handed one total, the plan charged the decode
///   chain's context and `o_proj` output at the whole wave's width: 8.7 MiB of
///   a 119.5 MiB span on a wave with no decode rows at all.
/// * **The forward phase does not scale with tokens.** Its head runs once per
///   *sequence*, so its cost is set by how many conversations the wave carries
///   and not by how long their prompts are. Priced at a row count it would be
///   absurd; priced at a constant — which is what `WAVE_FORWARD_BYTES` did —
///   it fits until the session count passes what the constant was measured on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WaveWidth {
    /// Tokens contributed by prefill groups.
    pub prefill_rows: usize,
    /// Tokens contributed by decode groups — one per sequence stepping, plus
    /// whatever a speculative block stages.
    pub decode_rows: usize,
    /// Rows the head produces logits for, which is what the forward phase
    /// scales with.
    ///
    /// Not the row count and not quite the sequence count: the head scores
    /// every decode row, the **last** token of each prefill sequence, and every
    /// row of a *verifying* span (each is a prediction to compare a proposal
    /// against). So an ordinary wave scores one row per sequence, and a
    /// speculative one scores a block per verifying sequence.
    pub scored_rows: usize,
    /// Rows a speculative **replay** stages onto the wave, and the spans it
    /// stages them for.
    ///
    /// A replay is not a wave in the ordinary sense: it re-runs the DeltaNet
    /// mixer to advance the recurrent state to an accepted prefix, and the
    /// tokens' logits were produced by the verify wave and are not recomputed.
    /// So it carves neither layer chain and no head — what it *does* carve is
    /// four staged operands and the two span tables built beside them, which is
    /// [`Chain::DeltaNetReplay`].
    ///
    /// Zero on every ordinary forward, which is what keeps that chain out of
    /// the attention phase's `max` everywhere but a replay.
    pub staged_rows: usize,
    /// Spans the replay stages — see [`Self::staged_rows`]. The span tables are
    /// per span, not per row.
    pub staged_spans: usize,
}

impl WaveWidth {
    /// Every token in the wave, whichever group contributed it.
    pub const fn rows(&self) -> usize {
        self.prefill_rows + self.decode_rows
    }

    /// A wave that is all prefill: `rows` tokens over `sequences` sequences,
    /// none of them verifying, so the head scores one row each.
    pub const fn prefill(rows: usize, sequences: usize) -> Self {
        Self {
            prefill_rows: rows,
            decode_rows: 0,
            scored_rows: sequences,
            staged_rows: 0,
            staged_spans: 0,
        }
    }

    /// A wave that is all decode: one row per sequence, every one of them
    /// scored.
    pub const fn decode(sequences: usize) -> Self {
        Self {
            prefill_rows: 0,
            decode_rows: sequences,
            scored_rows: sequences,
            staged_rows: 0,
            staged_spans: 0,
        }
    }

    /// A speculative **replay**: `rows` staged operand rows over `spans` spans,
    /// and nothing else.
    ///
    /// Every other unit is zero on purpose. A replay re-runs the mixer to
    /// advance the recurrent state and recomputes no logits, so it carves no
    /// attention chain, no FFN and no head — pricing those charged it for three
    /// phases it never touches. See [`Self::staged_rows`].
    pub const fn replay(rows: usize, spans: usize) -> Self {
        Self {
            prefill_rows: 0,
            decode_rows: 0,
            scored_rows: 0,
            staged_rows: rows,
            staged_spans: spans,
        }
    }

    /// `rows` more prefill tokens on top of this wave.
    pub const fn with_prefill(self, rows: usize) -> Self {
        Self {
            prefill_rows: self.prefill_rows + rows,
            ..self
        }
    }
}

/// The width the FFN carries its intermediates in, for activations of `act`.
///
/// An F16 activation is widened to BF16 for the SwiGLU, whose range can exceed
/// F16's; every other dtype is left alone. **The one definition** —
/// `forward_layer_batched_mixed` picks the FFN's dtype with it, and the plan
/// prices the casts around it with it, so the two cannot disagree about which
/// sessions have a cast at all.
pub fn ffn_work_dtype(act: DType) -> DType {
    if act == DType::F16 {
        DType::BF16
    } else {
        act
    }
}

/// A DeltaNet mixer's widths, as [`ModelGeometry::delta_net`] carries them.
///
/// Everything the mixer holds is F32 and stays F32: `S` is a running sum over
/// every token of a sequence, the one value in the stack with no bound on how
/// many additions it accumulates, and half precision drifts without bound in
/// context length — the opposite of what the O(1)-error design is for. So the
/// four projections ask the KO kernel to store F32 out of the accumulator it
/// already has, and the output projection reads that F32 directly. Only the
/// final `w_out` result narrows, to whatever the residual stream carries.
/// A **live** mixer's projections are downstream of a provenance break and
/// allocate from the CUDA pool, so none of these price a buffer on an ordinary
/// wave — see the note on the `DeltaNet*` variants. They price the
/// [`Chain::DeltaNetReplay`] instead: a speculative replay *stages* the same
/// four operands onto the wave deliberately (`spec::stage_on_wave`), because
/// the staged copy is the provenance root that keeps the replayed mixer off the
/// pool. So the widths are unused by the live chain and load-bearing for the
/// replayed one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaNetWidths {
    /// `2 · key_dim + value_dim` — the fused `[Q|K|V]` projection's width.
    pub conv_dim: usize,
    /// `head_dim × n_v_heads` — `z`'s width, and the mixer's output.
    pub value_dim: usize,
    /// V heads. `beta` and `alpha` are one F32 scalar per head per row.
    pub n_v_heads: usize,
}

/// A MoE layer's always-active **shared expert**, as
/// [`ModelGeometry::shared_expert`] carries it.
///
/// An ordinary SwiGLU every token goes through, scaled by a per-token
/// `sigmoid(w_gate · x)` and summed with the routed combine — Qwen3.5/3.6's
/// MoE. Only part of it lands on the span: the fused gate/up projection, the
/// SiLU, the product, the gate's tile-wide projection, its sigmoid, and the
/// final sum. The down projection goes through `forward_live_as`, whose int8 arm
/// quantizes its operand and so breaks provenance; its result, and the gated
/// product built on it, come off the CUDA pool and are not priced here.
/// Measured on Qwen3.5-35B-A3B at 2,100 rows: exactly those six carves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedExpertWidths {
    /// The shared SwiGLU's intermediate width.
    pub intermediate: usize,
    /// Columns the gate projection emits: its one real output, padded to a full
    /// KO tile so the int8 kernel can run it.
    pub gate_cols: usize,
}

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

/// Every buffer a layer allocates from the transient tier.
///
/// One variant per allocation site, and the list is the **union** over the
/// chains a phase can run — see the module header for why a union rather than a
/// maximum. Every one of these was read off [`super::wave_census`] on
/// Qwen3-30B-A3B; the comment on each says which chain allocates it, so a
/// reader can check the list against the code that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum WaveBuffer {
    /// Attention RMSNorm, in the encoding the QKV matmul consumes — q8a128 on
    /// an int8 session, `act_dtype` on a float one, per
    /// [`ModelGeometry::packed_norm`]. Both chains.
    AttnNorm,
    /// Q/K/V projection output, in the compute dtype.
    ///
    /// `qkv_cols` wide, which prices either dispatch: one segmented launch over
    /// the three KO weights narrowed afterwards, or three separate
    /// `forward_dynamic` calls whose widths sum to the same number.
    ///
    /// **There is no accumulate-dtype round trip to charge on either side of
    /// it.** The int8 path's MMA converts on the store out of registers, and the
    /// float path (`dense_qmatmul_float`) converts its activation only for a
    /// dtype outside `{F16, BF16, F32}` and otherwise stores at the activation's
    /// own width — so no session upcasts a compute-dtype operand to F32 and
    /// casts back. A pair of operand/accum charges modelling that round trip
    /// used to stand here and on `o_proj`, priced at `3 × hidden` and
    /// `qkv_cols` in F32: 90.2 MiB of a 219.9 MiB attention span on the 0.8B,
    /// for four buffers the census shows are never carved. What the census
    /// actually caught was a *DeltaNet* layer's F32 projections — a different
    /// layer kind sharing this arena, priced below at its own widths.
    QkvProjection,
    /// Q, copied out of the fused QKV buffer.
    ///
    /// The narrow is a strided view over a `qkv_cols`-wide row, so reshaping it
    /// to `[batch, seq, heads, dim]` cannot alias and copies. Both chains.
    QSplit,
    /// Q flattened to `[batch · heads · seq, dim]` for the head-wise RMSNorm.
    ///
    /// Free on the decode chain, where `seq == 1` leaves the transposed view
    /// contiguous; a real copy on prefill.
    QNormIn,
    /// Q RMSNorm output. Both chains.
    QNormOut,
    /// Q returned to `[batch, seq, heads · dim]` after the norm — a transpose
    /// then a reshape, so again a copy on prefill and free on decode.
    QHeadsPacked,
    /// The gate, split out of the interleaved `[q | gate]` projection. The
    /// narrow is strided over the `[.., heads, 2, dim]` view, so flattening it
    /// copies. Gated lineages only.
    GateSplit,
    /// `sigmoid(gate)`, materialised before the context multiply. Gated
    /// lineages only; the fused q8 decode context folds it into the kernel,
    /// but prefill and the FP decode path allocate it, and the plan prices
    /// the union.
    GateSigmoid,
    /// `context ⊙ sigmoid(gate)` — the gated context handed to `o_proj`.
    /// Gated lineages only, same union rule as [`Self::GateSigmoid`].
    GatedContext,
    /// Q re-ordered into the kernels' full-width rotary pairing — a gather
    /// copy (`RotaryLayout`). Partial-rotary models only.
    QRotaryPermute,
    /// K's half of the same re-ordering. Partial-rotary models only.
    KRotaryPermute,
    /// K, copied out of the fused QKV buffer. Both chains.
    KSplit,
    /// K flattened for the head-wise RMSNorm. Prefill only, as [`Self::QNormIn`].
    KNormIn,
    /// K RMSNorm output. Both chains.
    KNormOut,
    /// K returned to `[batch, seq, kv_heads · dim]`. Prefill only.
    KHeadsPacked,
    /// V, made contiguous out of the fused QKV buffer for the cache write.
    VContiguous,
    /// `q + bq`, on a stack with Q/K/V biases — see
    /// [`ModelGeometry::qkv_bias`]. Both chains, every width.
    QBias,
    /// `k + bk`, as [`Self::QBias`].
    KBias,
    /// `v + bv`, as [`Self::QBias`].
    VBias,
    /// The attention context in its **dense** form, which prefill and glue write.
    AttnOutput,
    /// The attention context in its **packed** form.
    ///
    /// The int8 decode kernel emits q8a1024 directly (`PagedDecode::q8_byte_size`)
    /// so `o_proj` needs no standalone quantize. Declared separately from
    /// [`Self::AttnOutput`] rather than folded into it because a mixed wave
    /// allocates both — the decode group's packed context and the prefill
    /// group's dense one — into the same generation.
    DecodeContext,
    /// `o_proj`'s result, in the compute dtype.
    ///
    /// Prefill's `o_proj` takes a `Float` context, the override quantizes it at
    /// the matmul, and that quantize breaks the provenance chain — the output
    /// lands on the pool. So this is a decode-chain carve, and a pure-prefill
    /// wave pays for it as part of the union.
    OProjOutput,

    // ── The DeltaNet mixer ──────────────────────────────────────────────────
    // The other thing a layer can open the attention generation for, on a
    // hybrid stack. Priced as its own chain and compared against attention's
    // with a `max`, not summed into it — see `Chain`.
    //
    // **Only part of this mixer is on the span, and the list below is that
    // part.** `QMatMul::forward_live_as` forks on the weight's int8 mode: the
    // int8 arm reaches the KO kernel through a standalone `to_dynamic`
    // quantize, and that quantize breaks the operand's provenance, so the
    // projection's result is allocated from the CUDA pool rather than from the
    // generation — and everything downstream of it (the causal conv, the
    // mixer's output, the norm-gate result, `w_out`) inherits the pool from its
    // operand and follows it off the span. The non-int8 arm has no quantize and
    // stays on the generation, paying one F32 upcast of its operand.
    //
    // So a DeltaNet layer's span cost is: the layer norm, plus one upcast and
    // one result for each projection whose weight is too small to have been
    // KO-repacked. Measured on the 0.8B at 2100 rows: five carves totalling
    // 21,772,800 B, which is what these five variants price to the byte.
    /// The DeltaNet layer's input norm — `ln1` over the packed buffer, which is
    /// where this chain is seeded on the span.
    DeltaNetNorm,
    /// `beta`'s operand, upcast to F32 by the float arm of `forward_live_as`.
    DeltaNetBetaOperand,
    /// `beta` itself: one F32 per V head per row.
    DeltaNetBetaProj,
    /// `alpha`'s operand upcast, a separate carve from [`Self::DeltaNetBetaOperand`]
    /// because the two projections each upcast the norm output for themselves.
    DeltaNetAlphaOperand,
    /// `alpha`, the same shape as [`Self::DeltaNetBetaProj`].
    DeltaNetAlphaProj,

    // ── A speculative replay's staged operands ──────────────────────────────
    // Measured on the 9B at 30 staged rows over 6 spans: six carves totalling
    // 1,482,480 B, every one of them `wave_empty` or a table built beside it.
    // The 27B is where leaving them undeclared stopped being survivable — 20
    // rows of `conv_dim` 10240 and `value_dim` 6144 come to 1,318,400 B against
    // a span priced at 1,216,768, and the replay exhausted it mid-flight.
    /// The stashed `[Q|K|V]` projection, staged onto the wave in F32.
    ReplayQkv,
    /// The stashed `z`, staged likewise.
    ReplayZ,
    /// The stashed `beta`, one F32 per V head per staged row.
    ReplayBeta,
    /// The stashed `alpha`, the same shape as [`Self::ReplayBeta`].
    ReplayAlpha,
    /// The span table's pointer block: four device pointers per span.
    ReplaySpanPtrs,
    /// The span table's extents: two `u32` per span.
    ReplaySpanExtents,

    /// FFN RMSNorm, in whatever encoding the expert GEMMs consume. Both
    /// dispatch paths, and priced dense for the reason given on
    /// [`Self::AttnNorm`].
    FfnNorm,
    /// The shared expert's fused `[gate | up]` projection, in `work_dtype`. It
    /// runs before the router — the routed half consumes the activation, so the
    /// shared half reads it first — and every shared buffer below is carved in
    /// that order. See [`SharedExpertWidths`] for which of its steps land here.
    SharedGateUp,
    /// `silu(gate)` over the shared expert's gate half.
    SharedAct,
    /// `silu(gate) ⊙ up` — what the shared down projection consumes.
    SharedGated,
    /// The shared expert's gate projection, a KO tile wide; only column 0 is
    /// the gate, the rest are the padding rows' zeros.
    SharedGateLogits,
    /// `sigmoid` of that column: one scalar per token, and the one shared
    /// buffer whose length is rarely a multiple of the alignment.
    SharedGateSigmoid,
    /// The router's per-token logits over every expert. Both paths.
    RouterLogits,
    /// Top-k routing weights, in F32. Both paths.
    RouteWeights,
    /// Top-k expert ids, as u32. Both paths.
    RouteIndices,
    /// The threaded pipeline's three assignment-indexed uploads, one `u32` per
    /// (token, expert) pair each: the gathered token ids, the weight ids in
    /// token-major order, and the permutation the scatter reads `down_out`
    /// through. Threaded pipeline only: the GPU-native path bucketizes into the
    /// dispatch tables' own workspace, off the span.
    ///
    /// **Three carves, not one table**, and the plan has to say so: each pays
    /// its own alignment pad when the assignment count is not a multiple of 64,
    /// where a single three-wide table would pay it once — 256 B under what the
    /// 35B carved at 2,100 rows, which is an overrun rather than a rounding.
    ///
    /// One upload per layer, because the pipeline computes every routed expert
    /// in one canonically-ordered call; splitting hits from misses made a
    /// token's k terms sum in a residency-dependent grouping. Measured at 9,880
    /// rows × 8 experts: three carves of 316,160 B. The plan used to charge
    /// eight tables per assignment, on the reasoning that batching made the
    /// count unknowable — 1.5 MiB a phase of ground conceded to tables that did
    /// not exist.
    RoutingTokenIds,
    /// See [`Self::RoutingTokenIds`].
    RoutingWeightIds,
    /// See [`Self::RoutingTokenIds`].
    RoutingPermutation,
    /// The per-token prefix offsets into the permutation, `rows + 1` of them.
    /// Threaded pipeline only.
    ///
    /// The one routing table whose length is not a multiple of the alignment,
    /// so the next carve pays a pad after it — the 156 B the census reports
    /// lost at 9,880 rows.
    RoutingTokenStarts,
    /// Tokens gathered into expert-major order for the grouped GEMMs.
    MoeGather,
    /// Gate projection over the gathered tokens.
    GateGemm,
    /// Up projection over the gathered tokens.
    UpGemm,
    /// Fused SwiGLU output, requantized to feed the down GEMM.
    SwigluAct,
    /// Down projection, in the accumulate dtype — which the scatter reads as it
    /// is. Both paths: `grouped_qmatmul_dev_q8a128` inherits the gathered
    /// operand's arena exactly as the pipeline's grouped GEMM does.
    ///
    /// **There is no cast after it.** The plan used to charge one, `experts ×
    /// hidden` in the compute dtype, that neither path makes —
    /// `fused_deterministic_scatter` validates the F32 operand rather than
    /// converting it. At 9,880 rows that was 323,747,840 B of a 1,816,655,360 B
    /// FFN plan that the census never saw carved.
    DownGemm,
    /// The MoE combine target the scatter accumulates into. Both paths.
    MoeCombine,
    /// `routed + gated shared` — the layer's output on a stack with a shared
    /// expert. It carves beside the routed combine because that is its first
    /// operand; the gated shared half it adds is on the pool.
    MoeSharedSum,

    // ── The dense FFN ───────────────────────────────────────────────────────
    // Four carves, measured on the 0.8B at 2100 rows: 62,630,400 B. No router,
    // no gather, no expert replication, and every intermediate in the compute
    // dtype — the down projection's own result is downstream of a quantize and
    // lands off the span, like the DeltaNet projections.
    /// The dense FFN's input norm.
    DenseFfnNorm,
    /// The fused `[gate | up]` projection: one GEMM, `2 × intermediate` wide.
    DenseGateUp,
    /// `silu(gate)`.
    DenseSilu,
    /// `silu(gate) ⊙ up` — the SwiGLU result the down projection consumes.
    DenseSwiglu,

    // ── The head, after the last layer ──────────────────────────────────────
    // The forward phase's whole content, and the reason it is sized in
    // sequences. Measured on the 0.8B: two carves, 497,920 B **per sequence**
    // — so `WAVE_FORWARD_BYTES`, at 16 MiB, covered 33 of them. This engine
    // composes waves of up to 64, where the phase needs 31.9 MiB and the
    // reservation would have been overrun by a forward that had already
    // launched every layer.
    /// The final norm, in the encoding the head's matmul consumes.
    HeadNorm,
    /// The float head-norm's F32 working copy, beside its `act_dtype` result.
    ///
    /// **Float sessions only.** An int8 session's `RmsNorm::forward_dynamic`
    /// emits q8a128 from one fused kernel and carves nothing else; the float arm
    /// goes through `forward_with_ticket`, whose RMSNorm materialises the row in
    /// F32 as well. Measured on Qwen2 at 60 scored rows, hidden 896: two carves,
    /// `60 × 896 × 2` and `60 × 896 × 4`, summing to the 322,560 B the arena
    /// peaked at.
    HeadNormF32,
    /// The head's logits: one row per scored row, `vocab` wide, in the compute
    /// dtype.
    ///
    /// **Int8 sessions only**, because only there do they land on the span. The
    /// int8 matmul carves its output from the operand's arena and converts on
    /// the store; the float arm reaches the dequantized-weight path through
    /// `to_owned_tensor`, which breaks provenance, so the logits come off the
    /// CUDA pool. Both were measured: Qwen3.5-0.8B (int8) carves
    /// `4 × 248,320 × 2` here, and Qwen2 (float) carves no logits at all.
    ///
    /// Being *returned* from the forward is not what decides it — the head's
    /// span is reset per forward precisely so the logits may outlive the layer
    /// guards — which is why this follows the session's mode and not the
    /// lifetime.
    HeadLogits,

    /// The MoE result **narrowed to the residual's dtype**, when the experts ran
    /// in a different one. Both paths.
    ///
    /// `ffn_forward` hands the experts `work_dtype` — BF16 for an F16 session,
    /// the F16-overflow stability cast — and then calls
    /// `to_dtype_mut(out_dtype)` on the combine. That returns early when the
    /// dtypes agree and **allocates** a fresh `rows × hidden` buffer when they do
    /// not: the census shows the `to_dtype_mut` carve on every F16 session and on
    /// no BF16 one.
    ///
    /// It used to be charged unconditionally, as a second combine target "for
    /// the threaded pipeline". Either path has exactly one combine target
    /// ([`Self::MoeCombine`]); the buffer beside it was always this cast, so a
    /// BF16 session paid `rows × hidden × 2` for it on every forward — while an
    /// F16 one was told the cast was free and the charge was for something else.
    MoeResultCast,
    /// The FFN norm's operand widened to `work_dtype` — the other half of the
    /// same F16 stability cast, on the MoE arm's **float** path.
    ///
    /// A packed operand is range-safe and skips it (`q8a128` carries its own
    /// scales), and a session whose `act_dtype` is already the work dtype has
    /// nothing to widen. So this is the one buffer that needs *both* a float
    /// session and an F16 activation.
    FfnNormOperand,
}

/// A run of buffers that one layer can allocate inside one generation.
///
/// The distinction a phase alone cannot make. Within a phase the plan charges
/// the **union** of the chains that can run — a mixed wave opens one attention
/// generation and runs its decode group and its prefill group inside it, so
/// both chains' buffers are live at once. That reasoning does not extend to a
/// hybrid's two *layer kinds*: a layer is an attention layer or a DeltaNet
/// layer, never both, and a generation is one layer's phase. Summing them would
/// price a wave for a layer that does not exist.
///
/// So a phase costs the **largest** of the chains that can open it, and each
/// chain costs the **sum** of its own buffers.
///
/// This was wrong in the other direction before it was named: the DeltaNet
/// chain was undeclared and `priced_intermediate` folded its conv width into
/// the *FFN* phase — which a DeltaNet layer runs identically to an attention
/// layer, and where its mixer carves nothing. That is 49.1 MiB of a 139.3 MiB
/// FFN span on the 0.8B, charged in a phase the buffer never appears in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum Chain {
    /// An attention layer's mixer: norm through `o_proj`.
    Attention,
    /// A DeltaNet layer's mixer: the four projections through `w_out`. Shares
    /// the attention generation, because it is the same scope — the thing a
    /// layer opens before its FFN.
    DeltaNet,
    /// A speculative **replay's** staged operands, in that same generation.
    ///
    /// The one chain that is not a *layer's*. `spec::stage_on_wave` copies the
    /// stashed `qkv`/`z`/`beta`/`alpha` onto the wave before replaying the
    /// mixer, deliberately: the staged copy is the provenance root that keeps
    /// the replayed chain off the pool. The two span tables are built
    /// `from_vec_beside` that copy, so they follow it onto the arena — which is
    /// also why they do **not** appear on an ordinary wave, where the anchor
    /// (`qkv`) is itself on the pool.
    DeltaNetReplay,
    /// The **MoE** FFN: the router, the expert gather, the grouped GEMMs and
    /// the combine.
    Ffn,
    /// The **dense** FFN: norm, one fused `[gate|up]` GEMM, SiLU, multiply.
    ///
    /// The same `max` relationship the two mixers have, one phase down. A layer
    /// dispatches to `QuantFfn::Dense` or `QuantFfn::Moe`, never both, and the
    /// two chains have almost nothing in common — the dense one has no router,
    /// no gather, no expert replication, and carries its intermediates in the
    /// compute dtype rather than the accumulate one. Pricing a dense model
    /// against the MoE list is not a margin, it is a different chain: on the
    /// 0.8B it over-charged `[gate|up]` by 30.1 MiB and *under*-charged the
    /// SwiGLU output by 21.6 MiB, and only passed because the two errors
    /// partly cancelled.
    DenseFfn,
    /// Per-forward setup, read from every layer.
    Forward,
}

impl Chain {
    /// The generation this chain allocates from.
    pub fn phase(&self) -> LayerPhase {
        match self {
            Self::Attention | Self::DeltaNet | Self::DeltaNetReplay => LayerPhase::Attention,
            Self::Ffn | Self::DenseFfn => LayerPhase::Ffn,
            Self::Forward => LayerPhase::Forward,
        }
    }
}

impl WaveBuffer {
    /// Which run of buffers this one belongs to.
    pub fn chain(&self) -> Chain {
        match self {
            Self::DeltaNetNorm
            | Self::DeltaNetBetaOperand
            | Self::DeltaNetBetaProj
            | Self::DeltaNetAlphaOperand
            | Self::DeltaNetAlphaProj => Chain::DeltaNet,
            Self::ReplayQkv
            | Self::ReplayZ
            | Self::ReplayBeta
            | Self::ReplayAlpha
            | Self::ReplaySpanPtrs
            | Self::ReplaySpanExtents => Chain::DeltaNetReplay,
            Self::DenseFfnNorm | Self::DenseGateUp | Self::DenseSilu | Self::DenseSwiglu => {
                Chain::DenseFfn
            }
            Self::HeadNorm | Self::HeadNormF32 | Self::HeadLogits => Chain::Forward,
            other => match other.phase() {
                LayerPhase::Attention => Chain::Attention,
                LayerPhase::Ffn => Chain::Ffn,
                LayerPhase::Forward => Chain::Forward,
            },
        }
    }

    /// When this buffer is live within its phase.
    pub fn phase(&self) -> LayerPhase {
        match self {
            Self::DeltaNetNorm
            | Self::DeltaNetBetaOperand
            | Self::DeltaNetBetaProj
            | Self::DeltaNetAlphaOperand
            | Self::DeltaNetAlphaProj
            | Self::ReplayQkv
            | Self::ReplayZ
            | Self::ReplayBeta
            | Self::ReplayAlpha
            | Self::ReplaySpanPtrs
            | Self::ReplaySpanExtents => LayerPhase::Attention,
            Self::AttnNorm
            | Self::QkvProjection
            | Self::QSplit
            | Self::QNormIn
            | Self::QNormOut
            | Self::QHeadsPacked
            | Self::GateSplit
            | Self::GateSigmoid
            | Self::GatedContext
            | Self::QRotaryPermute
            | Self::KRotaryPermute
            | Self::KSplit
            | Self::KNormIn
            | Self::KNormOut
            | Self::KHeadsPacked
            | Self::VContiguous
            | Self::QBias
            | Self::KBias
            | Self::VBias
            | Self::AttnOutput
            | Self::DecodeContext
            | Self::OProjOutput => LayerPhase::Attention,
            Self::FfnNorm
            | Self::SharedGateUp
            | Self::SharedAct
            | Self::SharedGated
            | Self::SharedGateLogits
            | Self::SharedGateSigmoid
            | Self::RouterLogits
            | Self::RouteWeights
            | Self::RouteIndices
            | Self::RoutingTokenIds
            | Self::RoutingWeightIds
            | Self::RoutingPermutation
            | Self::RoutingTokenStarts
            | Self::MoeGather
            | Self::GateGemm
            | Self::UpGemm
            | Self::SwigluAct
            | Self::DownGemm
            | Self::MoeCombine
            | Self::MoeSharedSum
            | Self::MoeResultCast
            | Self::FfnNormOperand
            | Self::DenseFfnNorm
            | Self::DenseGateUp
            | Self::DenseSilu
            | Self::DenseSwiglu => LayerPhase::Ffn,
            Self::HeadNorm | Self::HeadNormF32 | Self::HeadLogits => LayerPhase::Forward,
        }
    }

    /// This buffer's shape for a wave of width `w` — the declaration the call
    /// site and the admission gate both size from.
    ///
    /// **Each buffer names the unit it scales with**, which is the whole point
    /// of taking a [`WaveWidth`] rather than a row count: a decode-chain buffer
    /// is sized by the decode group, a prefill-chain one by the prefill group,
    /// and the head by the sequence count.
    ///
    /// Also exhaustive: a new variant must state its shape here, and takes its
    /// byte count from [`BufferShape::bytes`] rather than writing arithmetic of
    /// its own.
    pub fn shape(&self, g: &ModelGeometry, w: WaveWidth) -> BufferShape {
        let dense = |rows, cols, dtype| BufferShape {
            rows,
            cols,
            encoding: Encoding::Dense(dtype),
        };
        let q8 = |rows, cols| BufferShape {
            rows,
            cols,
            encoding: Encoding::Q8a128,
        };
        // Every row of the wave, which is what a buffer both chains allocate is
        // sized by. The two groups' own counts are named where they are used.
        let rows = w.rows();
        let er = g.expert_rows(rows);
        // Zero widths where there is no shared expert, which the guard arm below
        // prices at nothing before any of these are read.
        let shared = g.shared_expert.unwrap_or(SharedExpertWidths {
            intermediate: 0,
            gate_cols: 0,
        });
        // A layer norm's output encoding is the session's, not the buffer's —
        // the same fork for the attention norm, the FFN norm and the dense
        // FFN's norm.
        let norm = |rows: usize| {
            if g.packed_norm {
                q8(rows, g.hidden)
            } else {
                dense(rows, g.hidden, g.act_dtype)
            }
        };
        match self {
            Self::AttnNorm => norm(rows),
            Self::QkvProjection => dense(rows, g.qkv_cols(), g.act_dtype),
            // A bias add reads the narrowed view directly and writes its own
            // contiguous result, so where there are biases the adds are the Q,
            // K and V buffers and no split copy happens at any width.
            Self::QSplit | Self::KSplit | Self::VContiguous if g.qkv_bias => {
                dense(0, 0, g.act_dtype)
            }
            Self::QBias if g.qkv_bias => dense(rows, g.attn_cols(), g.act_dtype),
            Self::KBias | Self::VBias if g.qkv_bias => dense(rows, g.kv_cols(), g.act_dtype),
            Self::QBias | Self::KBias | Self::VBias => dense(0, 0, g.act_dtype),
            // **A one-row wave copies nothing out of the fused row.** The narrow
            // is along the last dimension, so it is strided only while some
            // leading dimension is wider than one. With a single row every
            // leading dimension is 1, `Shape::is_contiguous` skips unit
            // dimensions, and `contiguous()` hands the view straight back.
            // Measured on Qwen3-30B-A3B: a one-row decode priced 40,704 B of
            // attention and carved 30,464 — these three, exactly.
            //
            // Not Q on a gated lineage: its `[q | gate]` is interleaved per
            // head, so the narrow strides across heads however few rows there
            // are.
            Self::QSplit if rows == 1 && !g.gated_qkv => dense(0, 0, g.act_dtype),
            Self::KSplit | Self::VContiguous if rows == 1 => dense(0, 0, g.act_dtype),
            // Q is narrowed out of a wider buffer whenever there *is* one: a
            // fused projection's `qkv_cols` row, or a gated lineage's
            // interleaved `[q | gate]`. With neither, `wq`'s output is already
            // the tensor Q wants.
            Self::QSplit if g.fused_qkv || g.gated_qkv => dense(rows, g.attn_cols(), g.act_dtype),
            Self::QSplit => dense(0, 0, g.act_dtype),
            Self::QNormOut if g.head_qk_norm => dense(rows, g.attn_cols(), g.act_dtype),
            Self::QNormOut => dense(0, 0, g.act_dtype),
            // **Prefill's rows, not the wave's.** The reshapes exist because a
            // `[batch, seq, ..]` view cannot be transposed contiguously — which
            // `seq == 1` makes moot, so the decode chain carves none of them.
            // The decode census confirms it: nine carves, and these four are
            // not among them.
            Self::QNormIn | Self::QHeadsPacked if g.head_qk_norm && g.head_norm_reshapes => {
                dense(w.prefill_rows, g.attn_cols(), g.act_dtype)
            }
            Self::QNormIn | Self::QHeadsPacked => dense(0, 0, g.act_dtype),
            // The split happens on both chains — the gate is a strided narrow
            // out of the interleaved `[q | gate]` however the context is
            // computed.
            Self::GateSplit if g.gated_qkv => dense(rows, g.attn_cols(), g.act_dtype),
            // **The sigmoid and the apply are prefill's, on an int8 session.**
            // The fused q8 decode context folds the gate into the kernel, so a
            // decode row carves neither; the FP decode path computes them as
            // ordinary ops and does. This was the union the plan used to charge
            // at the whole wave's width — measured on the 9B at 20 decode rows,
            // exactly the 327,680 B its attention span went unused by.
            Self::GateSigmoid | Self::GatedContext if g.gated_qkv && g.packed_norm => {
                dense(w.prefill_rows, g.attn_cols(), g.act_dtype)
            }
            Self::GateSigmoid | Self::GatedContext if g.gated_qkv => {
                dense(rows, g.attn_cols(), g.act_dtype)
            }
            Self::GateSplit | Self::GateSigmoid | Self::GatedContext => dense(0, 0, g.act_dtype),
            Self::QRotaryPermute if g.partial_rotary => dense(rows, g.attn_cols(), g.act_dtype),
            Self::KRotaryPermute if g.partial_rotary => dense(rows, g.kv_cols(), g.act_dtype),
            Self::QRotaryPermute | Self::KRotaryPermute => dense(0, 0, g.act_dtype),
            // K and V are copied out of the fused row; separate projections
            // write them contiguous and neither copy exists.
            Self::KSplit | Self::VContiguous if g.fused_qkv => {
                dense(rows, g.kv_cols(), g.act_dtype)
            }
            Self::KSplit | Self::VContiguous => dense(0, 0, g.act_dtype),
            Self::KNormIn | Self::KHeadsPacked if g.head_qk_norm && g.head_norm_reshapes => {
                dense(w.prefill_rows, g.kv_cols(), g.act_dtype)
            }
            Self::KNormIn | Self::KHeadsPacked => dense(0, 0, g.act_dtype),
            Self::KNormOut if g.head_qk_norm => dense(rows, g.kv_cols(), g.act_dtype),
            Self::KNormOut => dense(0, 0, g.act_dtype),
            // The dense context is what the paged *prefill* kernel writes; the
            // decode kernel emits q8a1024 into `DecodeContext` instead. Two
            // buffers because a mixed wave carves both into one generation —
            // each at its own group's width, which is what a single row count
            // could not say.
            Self::AttnOutput => dense(w.prefill_rows, g.attn_cols(), g.act_dtype),
            // Only where the decode kernel emits it — see
            // `ModelGeometry::decode_q8_context`.
            Self::DecodeContext if g.decode_q8_context => q8(w.decode_rows, g.attn_cols()),
            Self::DecodeContext => dense(0, 0, g.act_dtype),
            Self::OProjOutput => dense(w.decode_rows, g.hidden, g.act_dtype),
            // The DeltaNet chain prices zero on a stack with no DeltaNet
            // layers, which makes `Chain::DeltaNet` sum to zero and drop out of
            // the phase's max — no all-attention model's span moves.
            Self::DeltaNetNorm => {
                let cols = if g.delta_net.is_some() { g.hidden } else { 0 };
                dense(rows, cols, g.act_dtype)
            }
            Self::DeltaNetBetaOperand | Self::DeltaNetAlphaOperand => {
                let cols = if g.delta_net.is_some() { g.hidden } else { 0 };
                dense(rows, cols, DType::F32)
            }
            Self::DeltaNetBetaProj | Self::DeltaNetAlphaProj => {
                dense(rows, g.delta_net.map_or(0, |d| d.n_v_heads), DType::F32)
            }
            // Sized by the **staged** rows and spans, which are zero on every
            // forward but a replay — so this chain drops out of the attention
            // phase's `max` everywhere else.
            Self::ReplayQkv => dense(
                w.staged_rows,
                g.delta_net.map_or(0, |d| d.conv_dim),
                DType::F32,
            ),
            Self::ReplayZ => dense(
                w.staged_rows,
                g.delta_net.map_or(0, |d| d.value_dim),
                DType::F32,
            ),
            Self::ReplayBeta | Self::ReplayAlpha => dense(
                w.staged_rows,
                g.delta_net.map_or(0, |d| d.n_v_heads),
                DType::F32,
            ),
            // Four device pointers and two extents per span — and nothing at
            // all on a stack with no mixer, which has no recurrent state, so no
            // rewind stash and no replay to build a table for.
            Self::ReplaySpanPtrs | Self::ReplaySpanExtents if g.delta_net.is_none() => {
                dense(0, 0, DType::U32)
            }
            Self::ReplaySpanPtrs => dense(w.staged_spans, 4, DType::I64),
            Self::ReplaySpanExtents => dense(w.staged_spans, 2, DType::U32),
            // The shared expert's half of the chain, on a stack whose MoE has
            // one — and nothing at all otherwise.
            Self::SharedGateUp
            | Self::SharedAct
            | Self::SharedGated
            | Self::SharedGateLogits
            | Self::SharedGateSigmoid
            | Self::MoeSharedSum
                if !g.is_moe() || g.shared_expert.is_none() =>
            {
                dense(0, 0, g.act_dtype)
            }
            Self::SharedGateUp => dense(rows, 2 * shared.intermediate, g.work_dtype()),
            Self::SharedAct | Self::SharedGated => dense(rows, shared.intermediate, g.work_dtype()),
            Self::SharedGateLogits => dense(rows, shared.gate_cols, g.work_dtype()),
            Self::SharedGateSigmoid => dense(rows, 1, g.work_dtype()),
            Self::MoeSharedSum => dense(rows, g.hidden, g.work_dtype()),
            // The two FFN chains are alternatives a layer dispatches between,
            // so each prices zero on the geometry that does not run it and the
            // phase's `max` takes whichever is live.
            Self::FfnNorm
            | Self::MoeCombine
            | Self::MoeResultCast
            | Self::FfnNormOperand
            | Self::DownGemm
            | Self::MoeGather
            | Self::RouterLogits
            | Self::RouteWeights
            | Self::RouteIndices
            | Self::RoutingTokenIds
            | Self::RoutingWeightIds
            | Self::RoutingPermutation
            | Self::RoutingTokenStarts
            | Self::GateGemm
            | Self::UpGemm
            | Self::SwigluAct
                if !g.is_moe() =>
            {
                dense(0, 0, g.act_dtype)
            }
            Self::FfnNorm => norm(rows),
            Self::RouterLogits => dense(rows, g.n_experts, g.act_dtype),
            Self::RouteWeights => dense(rows, g.experts_per_tok, DType::F32),
            Self::RouteIndices => dense(rows, g.experts_per_tok, DType::U32),
            Self::RoutingTokenIds | Self::RoutingWeightIds | Self::RoutingPermutation => {
                dense(er, 1, DType::U32)
            }
            // Prefix offsets: one per token plus the end, so none at all for a
            // phase no token reaches.
            Self::RoutingTokenStarts if rows == 0 => dense(0, 0, DType::U32),
            Self::RoutingTokenStarts => dense(rows + 1, 1, DType::U32),
            Self::MoeGather => q8(er, g.hidden),
            Self::GateGemm => dense(er, g.intermediate, g.accum_dtype),
            Self::UpGemm => dense(er, g.intermediate, g.accum_dtype),
            Self::SwigluAct => q8(er, g.intermediate),
            Self::DownGemm => dense(er, g.hidden, g.accum_dtype),
            Self::MoeCombine => dense(rows, g.hidden, g.act_dtype),
            Self::MoeResultCast if g.work_dtype() != g.act_dtype => {
                dense(rows, g.hidden, g.act_dtype)
            }
            Self::MoeResultCast => dense(0, 0, g.act_dtype),
            // The F16 stability cast's operand, which does not exist without
            // it: the experts run in `work_dtype`, which equals `act_dtype`
            // unless the session is F16. A packed operand skips it outright.
            Self::FfnNormOperand if g.work_dtype() != g.act_dtype && !g.packed_norm => {
                dense(rows, g.hidden, g.work_dtype())
            }
            Self::FfnNormOperand => dense(0, 0, g.act_dtype),
            Self::DenseFfnNorm | Self::DenseGateUp | Self::DenseSilu | Self::DenseSwiglu
                if g.is_moe() =>
            {
                dense(0, 0, g.act_dtype)
            }
            Self::DenseFfnNorm => norm(rows),
            Self::DenseGateUp => dense(rows, 2 * g.intermediate, g.act_dtype),
            Self::DenseSilu | Self::DenseSwiglu => dense(rows, g.intermediate, g.act_dtype),
            // One row per sequence, both of them — the head scores the last
            // token of each.
            // All three follow the **head's** mode, not a layer's — see
            // `packed_head`. A packed head carves a q8a128 norm and its logits;
            // a float one carves a dense norm and an F32 working copy, and its
            // logits leave the span.
            Self::HeadNorm if g.packed_head => q8(w.scored_rows, g.hidden),
            Self::HeadNorm => dense(w.scored_rows, g.hidden, g.act_dtype),
            Self::HeadNormF32 if !g.packed_head => dense(w.scored_rows, g.hidden, DType::F32),
            Self::HeadLogits if g.packed_head => dense(w.scored_rows, g.vocab, g.act_dtype),
            Self::HeadNormF32 | Self::HeadLogits => dense(0, 0, g.act_dtype),
        }
    }

    /// Bytes this buffer needs for a wave of width `w`.
    pub fn bytes(&self, g: &ModelGeometry, w: WaveWidth) -> usize {
        self.shape(g, w).bytes()
    }
}

/// Rows the width searches will consider before giving up on the budget.
///
/// The doubling in [`WavePlan::max_rows_for_tier`] and
/// [`WavePlan::max_rows_within`] climbs until a width does **not** fit, which
/// assumes a budget some width exceeds. A caller that prices against ground the
/// weight side might concede can hand in a budget no wave could spend, and then
/// the search runs until the row count overflows the cost arithmetic and panics
/// — a pure function faulting on a large argument, far from the caller that
/// chose it.
///
/// A million rows is four orders of magnitude past the widest wave this engine
/// composes (`MAX_PREFILL_TOKENS` is 8,192) and prices to tens of gigabytes, so
/// stopping here answers "wider than anything you can run" without ever
/// reaching the overflow.
const ROW_SEARCH_CEILING: usize = 1 << 20;

/// Prices a wave against the model's geometry.
///
/// Cheap to copy and free of interior state, so admission can hold one and call
/// it per candidate without synchronising anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavePlan {
    geometry: ModelGeometry,
}

impl WavePlan {
    pub fn new(geometry: ModelGeometry) -> Self {
        Self { geometry }
    }

    /// Bytes for one named buffer — what a call site asks before allocating, so
    /// that what it takes is what it was priced for.
    pub fn bytes(&self, buffer: WaveBuffer, w: WaveWidth) -> usize {
        buffer.bytes(&self.geometry, w)
    }

    /// What one phase needs: the **sum** of every buffer it allocates.
    ///
    /// The sum, not the peak, because a generation hands out memory from a bump
    /// cursor that only rewinds when the guard drops. A buffer that dies early
    /// still holds its bytes until the end of the phase, so the high-water mark
    /// of *live* buffers would under-price the span and the first wide wave
    /// would exhaust it.
    ///
    /// (Reusing a dead buffer's bytes needs offsets assigned from declared
    /// lifetimes rather than a cursor. That was built, measured at ~1.75x on a
    /// MoE layer, and removed: it required buffer to hand-declare which step it
    /// died on, with nothing tying the declaration to the code — and operand
    /// provenance made the whole planned-slot path unnecessary, since a site now
    /// inherits its arena from its input instead of looking up an offset.)
    ///
    /// Each buffer is charged one alignment, since each is a separately aligned
    /// range.
    ///
    /// **Summed within a chain, maximised across them.** A phase can be opened
    /// by more than one kind of layer — on a hybrid, an attention mixer or a
    /// DeltaNet one — and a generation holds exactly one of them, so the span is
    /// sized by the largest rather than by their total. See [`Chain`].
    pub fn phase_bytes(&self, phase: LayerPhase, w: WaveWidth) -> usize {
        Chain::iter()
            .filter(|c| c.phase() == phase)
            .map(|c| self.chain_bytes(c, w))
            .max()
            .unwrap_or(0)
    }

    /// What one chain's buffers cost, summed — see [`Self::phase_bytes`] for
    /// why a chain sums and a phase maximises.
    ///
    /// A buffer the geometry prices at zero is one this stack never allocates,
    /// so it takes no aligned range either and is charged nothing at all.
    ///
    /// **Walked as the cursor walks, not summed with each buffer rounded up.**
    /// A bump range starts at `aligned_start(base, cursor, align)` and then
    /// advances by its own length, so the alignment is paid on a buffer's
    /// *start* — which means the last buffer in a chain never pays for its
    /// tail, and a buffer whose length is already a multiple costs nothing
    /// extra at all.
    ///
    /// Rounding each length up instead is right for every chain whose shapes
    /// are 256-multiples — which is most of them, and why the census reports
    /// `0 B lost to alignment` there — and wrong wherever one is not. The
    /// replay's two span tables are 64 B and 16 B: charged rounded they cost
    /// 512, and the cursor spends 272.
    pub fn chain_bytes(&self, chain: Chain, w: WaveWidth) -> usize {
        WaveBuffer::iter()
            .filter(|b| b.chain() == chain)
            .map(|b| b.bytes(&self.geometry, w))
            .filter(|&len| len > 0)
            .fold(0usize, |cursor, len| {
                cursor.div_ceil(BUMP_ALIGNMENT) * BUMP_ALIGNMENT + len
            })
    }

    pub fn wave_bytes(&self, w: WaveWidth) -> usize {
        LayerPhase::iter()
            .map(|p| self.phase_bytes(p, w))
            .max()
            .unwrap_or(0)
    }

    /// What the transient tier will actually cost for `rows` — the **sum** of
    /// the phase spans, which is what `plan_wave_transient` buys.
    ///
    /// **Not [`Self::wave_bytes`], and that difference is the whole bug this
    /// exists to fix.** `wave_bytes` is the *max* over phases, because a single
    /// phase's span is what one layer needs at a time. The tier is every phase
    /// laid down side by side, so it costs their sum — roughly three times the
    /// max on this geometry. Sizing waves with the max under-priced the tier by
    /// that factor, and no choice of budget corrects a formula measuring the
    /// wrong quantity: measured, waves were composed whose tier wanted 6.5 GiB
    /// against a 912 MiB guarantee, and were refused 62 times in one run.
    ///
    /// **Mirrors the arithmetic in the forward exactly** — the two must agree
    /// or the wave admitted is not the wave priced. That includes the absence
    /// of padding: the forward buys `phase_bytes` for each of the three phases
    /// and nothing more, so a `TARGET_ARENA_BYTES` pad here (there were two,
    /// matching a `+ REGION_BYTES` the forward has since dropped) would refuse
    /// 32 MiB of waves the placement would have taken.
    ///
    /// **Rounded up to a whole region, because the placement is.** The tier is
    /// carved in regions (`place_transient` rounds its length to one), so a
    /// plan whose phases sum to 160.5 MiB stands 176 MiB tall. Priced unrounded,
    /// the scheduler saw a 163 MiB gap as enough for the least chunk, bought
    /// nothing, and the placement was refused by one region — on every wave,
    /// 29,000 times in seven minutes, the same wave re-formed each time because
    /// nothing in its price had changed.
    pub fn tier_bytes(&self, w: WaveWidth) -> usize {
        // `TARGET_ARENA_BYTES` is the region size read from the ungated source,
        // as `span_geometry` does — `region_pool::REGION_BYTES` is CUDA-only and
        // this plan is not.
        let raw = self.phase_bytes(LayerPhase::Attention, w)
            + self.phase_bytes(LayerPhase::Ffn, w)
            + self.phase_bytes(LayerPhase::Forward, w);
        raw.div_ceil(TARGET_ARENA_BYTES) * TARGET_ARENA_BYTES
    }

    /// The most **prefill** rows that can ride on top of `head` and still leave
    /// the whole tier inside `budget`.
    ///
    /// The bound a wave is actually composed against. `head` is what the caller
    /// has already put in the wave — its decode rows and its sequence count —
    /// because those price the tier too, and a search that ignored them handed
    /// back a width the placement then refused by the head.
    ///
    /// Bisected rather than divided: the `div_ceil` steps make the cost a
    /// staircase, so dividing the budget by a per-row average lands inside a
    /// step and over-admits.
    pub fn max_prefill_rows_for_tier(&self, budget: usize, head: WaveWidth) -> usize {
        if self.tier_bytes(head.with_prefill(1)) > budget {
            return 0;
        }
        let mut lo = 1usize;
        let mut hi = 2usize;
        while hi < ROW_SEARCH_CEILING && self.tier_bytes(head.with_prefill(hi)) <= budget {
            lo = hi;
            hi = hi.saturating_mul(2);
        }
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if self.tier_bytes(head.with_prefill(mid)) <= budget {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// The widest wave that fits in `budget` bytes — a **single phase's** bound.
    ///
    /// Not what a wave is sized by: the tier costs every phase together, which
    /// is [`Self::max_prefill_rows_for_tier`]. Kept for callers asking the
    /// narrower question of whether one phase's span holds a given width.
    ///
    /// Returns `0` when not even a single row fits, which admission must treat
    /// as a configuration error rather than as an empty wave: a budget that
    /// cannot price one token will never make progress, and silently admitting
    /// nothing would present as a hang.
    ///
    /// [`Self::wave_bytes`] is non-decreasing in `rows` (every term is a
    /// `div_ceil` of a product of it), so the largest fitting width is found by
    /// doubling to a bound that does not fit and bisecting. Exact rather than a
    /// closed form because the `div_ceil` steps make the cost a staircase, and
    /// dividing the budget by a per-row average would land inside a step and
    /// over-admit.
    pub fn max_rows_within(&self, budget: usize, head: WaveWidth) -> usize {
        if self.wave_bytes(head.with_prefill(1)) > budget {
            return 0;
        }
        let mut lo = 1usize;
        let mut hi = 2usize;
        while hi < ROW_SEARCH_CEILING && self.wave_bytes(head.with_prefill(hi)) <= budget {
            lo = hi;
            hi = hi.saturating_mul(2);
        }
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if self.wave_bytes(head.with_prefill(mid)) <= budget {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Whether a wave of width `w` fits in `budget`.
    pub fn fits(&self, w: WaveWidth, budget: usize) -> bool {
        self.wave_bytes(w) <= budget
    }

    /// Refuse a wave that does not fit, before any of it is assembled.
    ///
    /// The transient span already refuses an over-long bump, but by then the
    /// wave is part-built: earlier layers have launched, the half holds live
    /// ranges, and the failure names one buffer rather than the wave that was
    /// too wide. Pricing the whole wave up front turns that into one refusal at
    /// the gate, before any GPU work, with the width and the overage in the
    /// message.
    ///
    /// This is a hard error and it aborts the inference request. It is not a
    /// signal to trim the wave and continue: admission is supposed to have
    /// priced this wave against the same budget, so reaching here means the two
    /// disagree, and silently running a narrower wave would hide that
    /// disagreement for as long as it took to become a correctness bug.
    pub fn ensure_fits(&self, w: WaveWidth, budget: usize) -> candle::Result<()> {
        let cost = self.wave_bytes(w);
        if cost <= budget {
            return Ok(());
        }
        let head = WaveWidth {
            prefill_rows: 0,
            ..w
        };
        let widest = self.max_rows_within(budget, head);
        candle::bail!(
            "wave over budget: {} rows ({} prefill + {} decode, {} scored) need \
             {cost} B of transient span but the half holds {budget} B (over by {} B). \
             The widest prefill this budget admits beside that head is {widest} rows. \
             Admission priced this wave against the same plan, so this is an \
             accounting disagreement, not a wave to trim.\n{}",
            w.rows(),
            w.prefill_rows,
            w.decode_rows,
            w.scored_rows,
            cost - budget,
            self.describe(w)
        )
    }

    /// One line per phase and one per buffer within it — what to print when a
    /// span refuses, so an overflow names a shape rather than a number.
    pub fn describe(&self, w: WaveWidth) -> String {
        let mut out = format!(
            "wave plan @ {} prefill + {} decode rows, {} scored: {} B\n",
            w.prefill_rows,
            w.decode_rows,
            w.scored_rows,
            self.wave_bytes(w)
        );
        // Grouped by chain within the phase, because the phase is their max and
        // a flat list of buffers would not say which of them are alternatives.
        for phase in LayerPhase::iter() {
            out.push_str(&format!("  {phase:?}: {} B\n", self.phase_bytes(phase, w)));
            for chain in Chain::iter().filter(|c| c.phase() == phase) {
                out.push_str(&format!(
                    "    {chain:?}: {} B\n",
                    self.chain_bytes(chain, w)
                ));
                for buffer in WaveBuffer::iter().filter(|b| b.chain() == chain) {
                    out.push_str(&format!(
                        "      {buffer:?}: {} B\n",
                        buffer.bytes(&self.geometry, w)
                    ));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::WAVE_SPAN_BYTES;

    /// A prefill wave of `rows` tokens over one sequence.
    ///
    /// The shape most of these cases want: they vary one number and ask how the
    /// cost moves, and the prefill chain is the wider of the two. The cases that
    /// are *about* the split — the decode chain, the head — build their widths
    /// explicitly.
    fn w(rows: usize) -> WaveWidth {
        WaveWidth::prefill(rows, 1)
    }

    /// An empty wave, for the bounds that ask "how much prefill fits at all".
    /// The one sequence is the one the prefill rows will belong to.
    fn empty_head() -> WaveWidth {
        WaveWidth::prefill(0, 1)
    }

    /// **The tier costs the SUM of the phases, not the largest of them.**
    ///
    /// This is the defect the tier bound exists to fix, and nothing else would
    /// catch its return: sizing waves with `wave_bytes` under-prices the tier by
    /// roughly the phase count, which let waves ask for a 6.5 GiB tier against a
    /// 912 MiB guarantee and be refused 62 times in one run.
    #[test]
    fn tier_bytes_is_the_sum_of_phases_and_exceeds_the_largest_one() {
        let p = WavePlan::new(moe());
        for rows in [1usize, 64, 512, 4096] {
            assert!(
                p.tier_bytes(w(rows)) > p.wave_bytes(w(rows)),
                "rows {rows}: tier {} must exceed the largest single phase {}",
                p.tier_bytes(w(rows)),
                p.wave_bytes(w(rows)),
            );
        }
    }

    /// **The tier is priced in whole regions, as it is placed.** The placement
    /// rounds its length up to a region; a price that did not read a gap as
    /// enough when the placement then asked for one region more.
    #[test]
    fn tier_bytes_is_a_whole_number_of_regions() {
        let p = WavePlan::new(moe());
        for rows in [1usize, 64, 128, 512, 4096] {
            assert_eq!(
                p.tier_bytes(w(rows)) % TARGET_ARENA_BYTES,
                0,
                "rows {rows}: {} is not region-aligned",
                p.tier_bytes(w(rows)),
            );
        }
    }

    /// The bound is the inverse of the cost, and it must not overshoot: the
    /// widest accepted width fits, and one row more does not.
    #[test]
    fn max_prefill_rows_for_tier_is_exact_on_the_staircase() {
        let p = WavePlan::new(moe());
        let budget = WAVE_SPAN_BYTES;
        let head = WaveWidth::prefill(0, 1);
        let rows = p.max_prefill_rows_for_tier(budget, head);
        assert!(rows > 0, "the guaranteed span must price at least one row");
        assert!(
            p.tier_bytes(head.with_prefill(rows)) <= budget,
            "the accepted width must fit"
        );
        assert!(
            p.tier_bytes(head.with_prefill(rows + 1)) > budget,
            "one row more must not fit — otherwise the bound is leaving ground unused",
        );
    }

    /// **The head is charged, so a wave already carrying one gets less prefill.**
    /// A bound that ignored it handed back a width the placement then refused by
    /// exactly the head it had not been told about.
    #[test]
    fn a_standing_head_narrows_the_prefill_the_tier_admits() {
        let p = WavePlan::new(moe());
        let budget = WAVE_SPAN_BYTES;
        let empty = p.max_prefill_rows_for_tier(budget, WaveWidth::prefill(0, 1));
        let loaded = p.max_prefill_rows_for_tier(budget, WaveWidth::decode(32));
        assert!(
            loaded < empty,
            "32 decode rows must cost prefill width: {loaded} vs {empty}"
        );
    }

    /// A budget too small for a single row answers 0, which callers must treat
    /// as a configuration error rather than as an empty wave.
    #[test]
    fn a_budget_below_one_row_prices_nothing() {
        let p = WavePlan::new(moe());
        assert_eq!(p.max_prefill_rows_for_tier(1, WaveWidth::default()), 0);
    }

    /// Qwen3-30B-A3B's real shapes.
    fn moe() -> ModelGeometry {
        ModelGeometry {
            hidden: 2048,
            vocab: 151_936,
            intermediate: 768,
            n_head: 32,
            n_kv_head: 4,
            head_dim: 128,
            experts_per_tok: 8,
            n_experts: 128,
            // Every layer attends, so there is no mixer chain to compare
            // against and the attention chain sizes the phase alone.
            delta_net: None,
            act_dtype: DType::BF16,
            accum_dtype: DType::F32,
            // Every census these fixtures are pinned against was taken on an
            // int8 session, where the norm's fused epilogue emits q8a128.
            packed_norm: true,
            packed_head: true,
            gated_qkv: false,
            partial_rotary: false,
            shared_expert: None,
            qkv_bias: false,
            decode_q8_context: true,
            // The fused segmented projection and a `[batch, seq, ..]` wave —
            // the twelve carves `MEASURED_ATTN_PREFILL_PER_ROW` itemises.
            fused_qkv: true,
            head_qk_norm: true,
            head_norm_reshapes: true,
        }
    }

    /// Qwen3.5-0.8B's real shapes: gated `[q | gate]` projection and 64-of-256
    /// partial rotary. The head count is 8, read off the census (the q matmul
    /// emits `2 · 8 · 256` columns; an earlier fixture said 16 ungated — which
    /// priced the same `qkv_cols` by accident and hid the gate's whole
    /// downstream chain). The FFN width is the real one, 3584; the DeltaNet
    /// half of this hybrid is priced by its own widths, not folded in here.
    fn gated_partial_rotary() -> ModelGeometry {
        ModelGeometry {
            hidden: 1024,
            // The 0.8B's real vocabulary, read off the loaded checkpoint.
            vocab: 248_320,
            intermediate: 3584,
            n_head: 8,
            n_kv_head: 2,
            head_dim: 256,
            experts_per_tok: 1,
            n_experts: 1,
            // Three layers in four are DeltaNet on this lineage.
            delta_net: Some(DeltaNetWidths {
                conv_dim: 6144,
                value_dim: 2048,
                n_v_heads: 16,
            }),
            act_dtype: DType::BF16,
            accum_dtype: DType::F32,
            // Every census these fixtures are pinned against was taken on an
            // int8 session, where the norm's fused epilogue emits q8a128.
            packed_norm: true,
            packed_head: true,
            gated_qkv: true,
            partial_rotary: true,
            shared_expert: None,
            qkv_bias: false,
            decode_q8_context: true,
            // Three separate projections, and a wave already packed `[rows, ..]`
            // — so neither the K/V narrows nor the head-norm reshapes exist.
            fused_qkv: false,
            head_qk_norm: true,
            head_norm_reshapes: false,
        }
    }

    /// **Llama-3-8B's shapes, and the no-per-head-norm case.**
    ///
    /// Kept deliberately unlike the two above: Llama and Qwen2 have no
    /// `attn_q_norm`/`attn_k_norm` weight at all, so six attention buffers must
    /// price zero here. Charging them was six buffers those stacks never
    /// allocate, two of them `attn_cols` wide.
    fn dense() -> ModelGeometry {
        ModelGeometry {
            hidden: 4096,
            vocab: 128_256,
            intermediate: 12288,
            n_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            experts_per_tok: 1,
            n_experts: 1,
            delta_net: None,
            act_dtype: DType::BF16,
            accum_dtype: DType::F32,
            // Every census these fixtures are pinned against was taken on an
            // int8 session, where the norm's fused epilogue emits q8a128.
            packed_norm: true,
            packed_head: true,
            gated_qkv: false,
            partial_rotary: false,
            shared_expert: None,
            qkv_bias: false,
            decode_q8_context: true,
            fused_qkv: true,
            head_qk_norm: false,
            head_norm_reshapes: true,
        }
    }

    const MIB: usize = 1 << 20;

    /// Byte arithmetic lives in one function, so this is where it is pinned —
    /// against hand-computed values rather than against itself.
    #[test]
    fn dense_bytes_are_rows_times_cols_times_dtype() {
        let s = BufferShape {
            rows: 7,
            cols: 33,
            encoding: Encoding::Dense(DType::BF16),
        };
        assert_eq!(s.bytes(), 7 * 33 * 2);
    }

    #[test]
    fn q8a128_rounds_partial_groups_up_to_a_whole_super_block() {
        let q8 = |rows, cols| {
            BufferShape {
                rows,
                cols,
                encoding: Encoding::Q8a128,
            }
            .bytes()
        };
        assert_eq!(q8(1, 128), 1152, "one tile still costs a super-block");
        assert_eq!(q8(8, 128), 1152, "exactly eight tiles: one block");
        assert_eq!(q8(9, 128), 2304, "one tile over: a second block");
        assert_eq!(q8(1, 1), 1152, "a partial tile is a whole tile");
        // Two tiles, but a super-block holds eight — so still one block.
        assert_eq!(q8(1, 129), 1152, "129 columns is two tiles, one block");
        assert_eq!(q8(1, 128 * 9), 2304, "nine tiles crosses into a second");
    }

    /// Every variant must answer both questions for any real geometry. Iterates
    /// rather than lists, so a variant added later is covered untouched.
    #[test]
    fn every_buffer_has_a_phase_and_a_non_zero_size() {
        for g in [gated_partial_rotary(), moe(), dense()] {
            for rows in [1usize, 64, 4096] {
                for b in WaveBuffer::iter() {
                    // The gate chain, the rotary permutes and the DeltaNet
                    // mixer are the conditional shapes: each prices zero
                    // exactly when the geometry says the stack does not run it,
                    // and is charged in full whenever it does. Every other
                    // variant is unconditional, and a zero there is a variant
                    // nobody will notice is wrong.
                    let conditional = (matches!(
                        b,
                        WaveBuffer::GateSplit | WaveBuffer::GateSigmoid | WaveBuffer::GatedContext
                    ) && !g.gated_qkv)
                        || (matches!(b, WaveBuffer::QRotaryPermute | WaveBuffer::KRotaryPermute)
                            && !g.partial_rotary)
                        || (matches!(b, WaveBuffer::KSplit | WaveBuffer::VContiguous)
                            && (!g.fused_qkv || g.qkv_bias))
                        || (matches!(b, WaveBuffer::QBias | WaveBuffer::KBias | WaveBuffer::VBias)
                            && !g.qkv_bias)
                        || (matches!(b, WaveBuffer::DecodeContext) && !g.decode_q8_context)
                        || (matches!(b, WaveBuffer::QSplit) && !g.fused_qkv && !g.gated_qkv)
                        || (matches!(
                            b,
                            WaveBuffer::QNormIn
                                | WaveBuffer::QHeadsPacked
                                | WaveBuffer::KNormIn
                                | WaveBuffer::KHeadsPacked
                        ) && !(g.head_qk_norm && g.head_norm_reshapes))
                        || (matches!(b, WaveBuffer::QNormOut | WaveBuffer::KNormOut)
                            && !g.head_qk_norm)
                        || (matches!(b, WaveBuffer::FfnNormOperand)
                            && (g.work_dtype() == g.act_dtype || g.packed_norm))
                        // The result cast exists only where the experts ran in a
                        // wider dtype than the residual.
                        || (matches!(b, WaveBuffer::MoeResultCast)
                            && g.work_dtype() == g.act_dtype)
                        // The shared expert's half, on a stack whose MoE has one.
                        || (matches!(
                            b,
                            WaveBuffer::SharedGateUp
                                | WaveBuffer::SharedAct
                                | WaveBuffer::SharedGated
                                | WaveBuffer::SharedGateLogits
                                | WaveBuffer::SharedGateSigmoid
                                | WaveBuffer::MoeSharedSum
                        ) && g.shared_expert.is_none())
                        // The head's two alternatives: a packed session carves
                        // its logits on the span, a float one carries an F32
                        // working copy of the norm instead.
                        || (matches!(b, WaveBuffer::HeadNormF32) && g.packed_head)
                        || (matches!(b, WaveBuffer::HeadLogits) && !g.packed_head)
                        || ((b.chain() == Chain::DeltaNet
                            || b.chain() == Chain::DeltaNetReplay)
                            && g.delta_net.is_none())
                        || (b.chain() == Chain::Ffn && !g.is_moe())
                        || (b.chain() == Chain::DenseFfn && g.is_moe());
                    // Every unit non-zero, so a buffer sized by any of the three
                    // is exercised and a zero is a defect rather than a width
                    // this case happened not to supply.
                    let width = WaveWidth {
                        prefill_rows: rows,
                        decode_rows: rows,
                        scored_rows: rows,
                        staged_rows: rows,
                        staged_spans: rows,
                    };
                    let s = b.shape(&g, width);
                    if conditional {
                        assert_eq!(b.bytes(&g, width), 0, "{b:?} priced while disabled");
                        continue;
                    }
                    assert!(b.bytes(&g, width) > 0, "{b:?} sized zero at {rows} rows");
                    assert!(s.rows > 0 && s.cols > 0, "{b:?} has an empty shape");
                    let _ = b.phase();
                }
            }
        }
    }

    /// The gated attention chain, pinned **carve for carve** against
    /// a `wave-census-labels` build on the 0.8B at its peak attention generation —
    /// 2100 rows, thirteen carves totalling 88,435,200 B.
    ///
    /// Stated as the measurement rather than as the plan's own arithmetic,
    /// which is the only form that can catch the error it replaces: four
    /// buffers modelling an F32 projection round trip used to be declared here,
    /// pinned at a different census, and they priced 90.2 MiB of a 219.9 MiB
    /// span against a generation that carves no F32 at all. The eleven
    /// assertions below are the eleven carves the census actually shows.
    #[test]
    fn the_gated_chain_prices_the_measured_carves() {
        let g = gated_partial_rotary();
        let plan = WavePlan::new(g);
        // The census's wave was all prefill over four contexts.
        let rows = WaveWidth::prefill(2100, 4);
        // The three projections — `[q|gate]`, K, V — dispatched separately but
        // summing to the one `qkv_cols` charge.
        assert_eq!(
            WaveBuffer::QkvProjection.bytes(&g, rows),
            17_203_200 + 2_150_400 + 2_150_400,
            "the gate-widened Q projection plus K and V, all in the compute dtype"
        );
        for (b, want) in [
            (WaveBuffer::QSplit, 8_601_600),
            (WaveBuffer::QNormOut, 8_601_600),
            (WaveBuffer::KNormOut, 2_150_400),
            (WaveBuffer::QRotaryPermute, 8_601_600),
            (WaveBuffer::KRotaryPermute, 2_150_400),
            (WaveBuffer::GateSplit, 8_601_600),
            (WaveBuffer::GateSigmoid, 8_601_600),
            (WaveBuffer::GatedContext, 8_601_600),
            (WaveBuffer::AttnOutput, 8_601_600),
        ] {
            assert_eq!(b.bytes(&g, rows), want, "{b:?} against its measured carve");
        }
        // The six copies a fused projection and a `[batch, seq, ..]` wave force,
        // and that this chain has neither of. They are not "margin" — they are
        // buffers another model allocates and this one does not, and charging
        // them here was 25.8 MiB of a 119.5 MiB span.
        for b in [
            WaveBuffer::KSplit,
            WaveBuffer::VContiguous,
            WaveBuffer::QNormIn,
            WaveBuffer::QHeadsPacked,
            WaveBuffer::KNormIn,
            WaveBuffer::KHeadsPacked,
        ] {
            assert_eq!(b.bytes(&g, rows), 0, "{b:?} is not carved on this chain");
        }
        // **The whole chain, to the byte**, against the thirteen carves the
        // census itemises — and this is the assertion the width split exists
        // for. `DecodeContext` and `OProjOutput` are the decode group's; a wave
        // with no decode rows carves neither, and priced at one total row count
        // they cost this span 8.7 MiB it could never spend.
        assert_eq!(
            plan.chain_bytes(Chain::Attention, rows),
            88_435_200,
            "a pure-prefill wave must price its measured generation exactly"
        );
        for b in [WaveBuffer::DecodeContext, WaveBuffer::OProjOutput] {
            assert_eq!(
                b.bytes(&g, rows),
                0,
                "{b:?} charged on a wave with no decode rows"
            );
            assert!(
                b.bytes(&g, WaveWidth::decode(2100)) > 0,
                "{b:?} must still be charged when there ARE decode rows"
            );
        }

        // A geometry with neither flag is charged nothing for the gate chain or
        // the permutes, so no ungated model's span moves.
        let plain = ModelGeometry {
            gated_qkv: false,
            partial_rotary: false,
            ..g
        };
        for b in [
            WaveBuffer::GateSplit,
            WaveBuffer::GateSigmoid,
            WaveBuffer::GatedContext,
            WaveBuffer::QRotaryPermute,
            WaveBuffer::KRotaryPermute,
        ] {
            assert_eq!(b.bytes(&plain, rows), 0, "{b:?} priced while disabled");
        }
    }

    /// The totals must account for every variant exactly once — the property
    /// that breaks if a phase filter is ever written by hand.
    #[test]
    fn the_phases_partition_every_buffer() {
        let counted: usize = LayerPhase::iter()
            .map(|p| WaveBuffer::iter().filter(|b| b.phase() == p).count())
            .sum();
        assert_eq!(counted, WaveBuffer::iter().count());
    }

    /// The same for chains, and additionally that the two agree: a buffer's
    /// phase must be its chain's phase, or `phase_bytes` maximises over a
    /// partition that does not cover the phase it is asked about.
    #[test]
    fn the_chains_partition_every_buffer_and_agree_with_its_phase() {
        let counted: usize = Chain::iter()
            .map(|c| WaveBuffer::iter().filter(|b| b.chain() == c).count())
            .sum();
        assert_eq!(counted, WaveBuffer::iter().count());
        for b in WaveBuffer::iter() {
            assert_eq!(
                b.chain().phase(),
                b.phase(),
                "{b:?} is in chain {:?} (phase {:?}) but reports phase {:?}",
                b.chain(),
                b.chain().phase(),
                b.phase()
            );
        }
    }

    /// **The DeltaNet chain against its measured generation, to the byte.**
    ///
    /// A `wave-census-labels` build on the 0.8B at 2100 rows: five carves,
    /// 21,772,800 B, `0 B lost to alignment`. That total is the assertion — a chain priced
    /// from a list of shapes is only as good as the list, and the sum is what
    /// says nothing was left off it and nothing imagined onto it.
    ///
    /// It is five carves and not eleven because `forward_live_as` sends the two
    /// KO-repacked projections through a standalone quantize that breaks
    /// provenance, taking them and everything downstream off the span — see the
    /// note on the DeltaNet variants.
    #[test]
    fn the_delta_net_chain_prices_its_measured_generation() {
        let g = gated_partial_rotary();
        let plan = WavePlan::new(g);
        let rows = WaveWidth::prefill(2100, 4);
        for (b, want) in [
            (WaveBuffer::DeltaNetNorm, 4_300_800),
            (WaveBuffer::DeltaNetBetaOperand, 8_601_600),
            (WaveBuffer::DeltaNetBetaProj, 134_400),
            (WaveBuffer::DeltaNetAlphaOperand, 8_601_600),
            (WaveBuffer::DeltaNetAlphaProj, 134_400),
        ] {
            assert_eq!(b.bytes(&g, rows), want, "{b:?} against its measured carve");
        }
        assert_eq!(
            plan.chain_bytes(Chain::DeltaNet, rows),
            21_772_800,
            "the whole generation, alignment included — the census measured no \
             alignment loss at all, so the chain must price the bare sum"
        );
    }

    /// **The dense FFN chain against its measured generation, to the byte.**
    ///
    /// Four carves on the 0.8B at 2100 rows, `0 B lost to alignment`:
    /// `62,630,400 B` against an intermediate of 3584. The `[gate|up]` GEMM is
    /// one fused launch `2 × intermediate` wide, and everything after the norm
    /// stays in the compute dtype.
    #[test]
    fn the_dense_ffn_chain_prices_its_measured_generation() {
        let g = gated_partial_rotary();
        let plan = WavePlan::new(g);
        let rows = WaveWidth::prefill(2100, 4);
        for (b, want) in [
            (WaveBuffer::DenseGateUp, 30_105_600),
            (WaveBuffer::DenseSilu, 15_052_800),
            (WaveBuffer::DenseSwiglu, 15_052_800),
        ] {
            assert_eq!(b.bytes(&g, rows), want, "{b:?} against its measured carve");
        }
        // The norm is packed, because this census was taken on an int8 session
        // and its RMSNorm fuses the quantize into its epilogue. A float
        // session's norm is the compute dtype and 1.8× larger — a real
        // difference in what the span holds, which is why the geometry carries
        // the mode rather than pricing the larger of the two.
        assert_eq!(WaveBuffer::DenseFfnNorm.bytes(&g, rows), 2_419_200);
        assert_eq!(
            WaveBuffer::DenseFfnNorm.bytes(
                &ModelGeometry {
                    packed_norm: false,
                    ..g
                },
                rows
            ),
            4_300_800,
            "a float session's norm is dense"
        );
        assert_eq!(
            plan.chain_bytes(Chain::DenseFfn, rows),
            62_630_400,
            "the measured generation, to the byte"
        );
        // A MoE geometry prices the dense chain at nothing, and a dense one
        // prices the expert pipeline at nothing — they never both run.
        assert_eq!(WavePlan::new(moe()).chain_bytes(Chain::DenseFfn, rows), 0);
        assert_eq!(plan.chain_bytes(Chain::Ffn, rows), 0);
    }

    /// **A stack with no per-head Q/K norm is charged for none of it.**
    ///
    /// Llama and Qwen2 have no `attn_q_norm`/`attn_k_norm` weight, so six
    /// attention buffers — the two norm outputs and the four reshape copies
    /// around them — must price zero. They were charged unconditionally, which
    /// on Llama-3-8B's shapes is `rows × (attn_cols + kv_cols) × 2` of span
    /// those stacks can never spend.
    #[test]
    fn a_stack_without_per_head_norms_pays_for_none_of_them() {
        let rows = WaveWidth::prefill(1000, 1);
        let without = dense();
        assert!(!without.head_qk_norm);
        for b in [
            WaveBuffer::QNormIn,
            WaveBuffer::QNormOut,
            WaveBuffer::QHeadsPacked,
            WaveBuffer::KNormIn,
            WaveBuffer::KNormOut,
            WaveBuffer::KHeadsPacked,
        ] {
            assert_eq!(b.bytes(&without, rows), 0, "{b:?} charged without a norm");
        }
        // And the same geometry *with* one pays for all six, so the flag is
        // doing the work rather than some other term happening to be zero.
        let with = ModelGeometry {
            head_qk_norm: true,
            ..without
        };
        for b in [
            WaveBuffer::QNormIn,
            WaveBuffer::QNormOut,
            WaveBuffer::QHeadsPacked,
            WaveBuffer::KNormIn,
            WaveBuffer::KNormOut,
            WaveBuffer::KHeadsPacked,
        ] {
            assert!(b.bytes(&with, rows) > 0, "{b:?} not charged with a norm");
        }
    }

    /// **A float session narrows nothing out of a fused projection, because it
    /// has no fused projection.**
    ///
    /// `project_qkv` forks on the operand: `Int8` takes `qkv_segmented` and
    /// three narrows that copy; `Float` takes three separate matmuls whose
    /// outputs are already contiguous. Hard-coding the fused case charged every
    /// float session three copies it never makes — and `QkvProjection` already
    /// prices the widths, which are identical either way.
    #[test]
    fn a_float_session_is_not_charged_the_fused_projections_narrows() {
        let rows = WaveWidth::prefill(1000, 1);
        let packed = moe();
        let float = ModelGeometry {
            packed_norm: false,
            fused_qkv: false,
            ..packed
        };
        for b in [
            WaveBuffer::QSplit,
            WaveBuffer::KSplit,
            WaveBuffer::VContiguous,
        ] {
            assert!(b.bytes(&packed, rows) > 0, "{b:?} missing when fused");
            assert_eq!(b.bytes(&float, rows), 0, "{b:?} charged when not fused");
        }
        // The projection widths themselves do not move — one launch or three,
        // the same columns are written.
        assert_eq!(
            WaveBuffer::QkvProjection.bytes(&packed, rows),
            WaveBuffer::QkvProjection.bytes(&float, rows),
        );
    }

    /// **The F16 stability cast's operand is charged only where it exists.**
    ///
    /// The MoE experts run in `work_dtype`, which is `act_dtype` everywhere
    /// except F16 — where the SwiGLU intermediates are widened to BF16 for
    /// range. That widening is a real `to_dtype` on the float arm and nothing
    /// at all on the packed arm, where q8a128 carries its own scales.
    ///
    /// The cast on the way *back* out is `to_dtype_mut`, which returns early
    /// when the dtypes agree and **allocates** when they do not — the census
    /// shows its carve on every F16 session and on no BF16 one. So it is
    /// charged exactly where `work_dtype` differs from `act_dtype`, and nowhere
    /// else. It was once declared as never allocating, and charged anyway under
    /// another name at a size that happened to match.
    #[test]
    fn the_f16_stability_cast_is_charged_only_where_it_exists() {
        let rows = WaveWidth::prefill(1000, 1);
        let bf16 = moe();
        assert_eq!(bf16.work_dtype(), bf16.act_dtype);
        assert_eq!(WaveBuffer::FfnNormOperand.bytes(&bf16, rows), 0);
        assert_eq!(WaveBuffer::MoeResultCast.bytes(&bf16, rows), 0);

        let f16 = ModelGeometry {
            act_dtype: DType::F16,
            ..bf16
        };
        assert_eq!(f16.work_dtype(), DType::BF16);
        // Still nothing: a packed operand is range-safe and skips the widening.
        assert_eq!(WaveBuffer::FfnNormOperand.bytes(&f16, rows), 0);
        // But the result comes back in BF16 either way, and narrowing it to the
        // F16 residual is a real buffer.
        assert_eq!(WaveBuffer::MoeResultCast.bytes(&f16, rows), 1000 * 2048 * 2);
        let f16_float = ModelGeometry {
            packed_norm: false,
            ..f16
        };
        assert!(WaveBuffer::FfnNormOperand.bytes(&f16_float, rows) > 0);
    }

    /// **One combine target, whichever path dispatches the experts.**
    ///
    /// Both paths scatter into a single `rows × hidden` buffer. The plan used to
    /// charge a second one for "the threaded pipeline" — which was really the
    /// result cast above, so every BF16 session paid for a buffer it never made.
    #[test]
    fn a_moe_layer_has_one_combine_target() {
        let rows = WaveWidth::prefill(1000, 1);
        assert_eq!(WaveBuffer::MoeCombine.bytes(&moe(), rows), 1000 * 2048 * 2);
        // None on a dense stack, which has no expert dispatch at all.
        for b in [WaveBuffer::MoeCombine, WaveBuffer::MoeResultCast] {
            assert_eq!(b.bytes(&dense(), rows), 0);
        }
    }

    /// **The FFN phase is the census, to the byte**, at two widths and both
    /// session dtypes — routing tables, the result cast and the alignment pad
    /// after the token offsets included.
    ///
    /// The plan charged 325,288,960 B more than the F16 generation carved: a
    /// down-projection cast neither dispatch path makes, and routing tables
    /// bounded at eight per assignment against the three the pipeline uploads.
    #[test]
    fn the_ffn_phase_is_the_measured_generation() {
        let bf16 = WavePlan::new(moe());
        let width = WaveWidth::prefill(4960, 20);
        assert_eq!(
            bf16.phase_bytes(LayerPhase::Ffn, width),
            MEASURED_FFN_BF16_4960,
            "{}",
            bf16.describe(width)
        );
        let f16 = WavePlan::new(ModelGeometry {
            act_dtype: DType::F16,
            ..moe()
        });
        let width = WaveWidth::prefill(9880, 20);
        assert_eq!(
            f16.phase_bytes(LayerPhase::Ffn, width),
            MEASURED_FFN_F16_9880,
            "{}",
            f16.describe(width)
        );
    }

    /// **A shared expert's carves are priced, and only the ones on the span.**
    /// Its down projection and the gated product built on it come off the
    /// pool; charging them would be slack, omitting the six that are here was
    /// an overrun the down-cast overcharge happened to hide.
    #[test]
    fn the_shared_expert_ffn_is_the_measured_generation() {
        let bf16 = WavePlan::new(moe_35b());
        let width = WaveWidth::prefill(2100, 4);
        assert_eq!(
            bf16.phase_bytes(LayerPhase::Ffn, width),
            MEASURED_FFN_35B_BF16_2100,
            "{}",
            bf16.describe(width)
        );
        let f16 = WavePlan::new(ModelGeometry {
            act_dtype: DType::F16,
            ..moe_35b()
        });
        let width = WaveWidth::prefill(1070, 2);
        assert_eq!(
            f16.phase_bytes(LayerPhase::Ffn, width),
            MEASURED_FFN_35B_F16_1070,
            "{}",
            f16.describe(width)
        );
        // And none of it on a MoE without a shared expert.
        for b in [
            WaveBuffer::SharedGateUp,
            WaveBuffer::SharedAct,
            WaveBuffer::SharedGated,
            WaveBuffer::SharedGateLogits,
            WaveBuffer::SharedGateSigmoid,
            WaveBuffer::MoeSharedSum,
        ] {
            assert_eq!(b.bytes(&moe(), width), 0, "{b:?}");
            assert!(b.bytes(&moe_35b(), width) > 0, "{b:?}");
        }
    }

    /// **The forward phase against its measured generation, and against the
    /// constant it replaces.**
    ///
    /// Two carves on the 0.8B, `0 B lost to alignment`: the packed head norm at
    /// 1,152 B a scored row and the logits at 496,640 B (248,320 vocab × BF16).
    /// 1,991,168 B at four scored rows, and 497,920 B at one.
    ///
    /// The second half is the reason this matters beyond slack.
    /// `WAVE_FORWARD_BYTES` is 16 MiB, which covers 33 scored rows — and this
    /// engine composes waves of 64 sessions. Past that the phase needed more
    /// than its reservation, and the span would have exhausted *after* every
    /// layer had launched, as a refusal with nothing in it naming the head.
    #[test]
    fn the_forward_phase_prices_its_measured_generation_and_outgrows_the_old_constant() {
        let g = gated_partial_rotary();
        let plan = WavePlan::new(g);
        assert_eq!(
            plan.chain_bytes(Chain::Forward, WaveWidth::prefill(2100, 4)),
            1_991_168,
            "the measured generation at four scored rows, to the byte"
        );
        assert_eq!(
            plan.chain_bytes(Chain::Forward, WaveWidth::prefill(2100, 1)),
            497_920,
            "and at one"
        );
        // It scales with scored rows and not with tokens — the whole reason the
        // forward phase is sized separately from the layer phases.
        assert_eq!(
            plan.chain_bytes(Chain::Forward, WaveWidth::prefill(8192, 4)),
            plan.chain_bytes(Chain::Forward, WaveWidth::prefill(2100, 4)),
        );
        // The constant this replaces, and the width at which it stopped being a
        // reservation and started being a ceiling.
        const OLD_CONSTANT: usize = 16 << 20;
        let at_33 = plan.chain_bytes(Chain::Forward, WaveWidth::prefill(1, 33));
        let at_64 = plan.chain_bytes(Chain::Forward, WaveWidth::prefill(1, 64));
        assert!(at_33 <= OLD_CONSTANT, "33 sessions fitted the constant");
        assert!(
            at_64 > OLD_CONSTANT,
            "64 sessions need {at_64} B against the {OLD_CONSTANT} B the constant reserved"
        );
    }

    /// **The head is two different chains, and the session's mode picks one.**
    ///
    /// Both measured. An **int8** session carves a packed norm and its logits:
    /// Qwen3.5-0.8B at 4 scored rows, `4 × 1,152 + 4 × 248,320 × 2`. A **float**
    /// session carves a dense norm and its F32 working copy, and no logits at
    /// all — the dequantized-weight path reaches the matmul through
    /// `to_owned_tensor`, which breaks provenance and puts the result on the
    /// pool. Qwen2 at 60 scored rows, hidden 896: `60 × 896 × 2 + 60 × 896 × 4`
    /// = 322,560 B, which is exactly what its forward arena peaked at.
    ///
    /// Charging the int8 shape to a float session was 17.1 MiB of a 17.4 MiB
    /// span — the phase was 98% slack, and every byte of it logits that were
    /// never there.
    #[test]
    fn the_head_is_priced_for_the_session_it_runs_in() {
        let int8 = gated_partial_rotary();
        assert!(int8.packed_norm);
        let plan = WavePlan::new(int8);
        let w = WaveWidth::prefill(2100, 4);
        assert_eq!(WaveBuffer::HeadNormF32.bytes(&int8, w), 0);
        assert_eq!(WaveBuffer::HeadLogits.bytes(&int8, w), 4 * 248_320 * 2);
        assert_eq!(plan.chain_bytes(Chain::Forward, w), 1_991_168);

        // Qwen2's shapes, on the float path its gate actually runs — and note
        // `packed_norm` stays **true**. That is the measured case: its layers
        // are packed and its head is not, so the head follows `packed_head`
        // alone. Pricing it from the layers' flag was the whole 17.1 MiB.
        let float = ModelGeometry {
            packed_head: false,
            hidden: 896,
            vocab: 151_936,
            act_dtype: DType::F16,
            ..int8
        };
        assert!(float.packed_norm, "the layers stay packed");
        let w60 = WaveWidth::prefill(4096, 60);
        assert_eq!(
            WaveBuffer::HeadLogits.bytes(&float, w60),
            0,
            "a float session's logits leave the span"
        );
        assert_eq!(WaveBuffer::HeadNorm.bytes(&float, w60), 60 * 896 * 2);
        assert_eq!(WaveBuffer::HeadNormF32.bytes(&float, w60), 60 * 896 * 4);
        assert_eq!(
            WavePlan::new(float).chain_bytes(Chain::Forward, w60),
            322_560,
            "the measured generation, to the byte"
        );
    }

    /// **A speculative replay's staged operands, against its measured
    /// generation.**
    ///
    /// A `wave-census-labels` build on Qwen3.5-9B: six carves, 1,482,480 B, at 30
    /// staged rows over 6 spans — `conv_dim` 8192 and `value_dim` 4096 in F32,
    /// two `n_v_heads`-wide scalars, and the span table's 4 pointers and 2
    /// extents per span.
    ///
    /// **And it is charged to nothing else.** The chain is sized by
    /// `staged_rows`/`staged_spans`, which are zero on every ordinary forward,
    /// so it drops out of the attention phase's `max` everywhere but a replay.
    /// Left undeclared it fit under the over-charge on the 9B and did not on
    /// the 27B, where the replay exhausted the span mid-flight.
    #[test]
    fn the_replay_chain_prices_its_measured_generation() {
        let g = ModelGeometry {
            delta_net: Some(DeltaNetWidths {
                conv_dim: 8192,
                value_dim: 4096,
                n_v_heads: 32,
            }),
            ..gated_partial_rotary()
        };
        let plan = WavePlan::new(g);
        let replay = WaveWidth::replay(30, 6);
        for (b, want) in [
            (WaveBuffer::ReplayQkv, 30 * 8192 * 4),
            (WaveBuffer::ReplayZ, 30 * 4096 * 4),
            (WaveBuffer::ReplayBeta, 30 * 32 * 4),
            (WaveBuffer::ReplayAlpha, 30 * 32 * 4),
            (WaveBuffer::ReplaySpanPtrs, 6 * 4 * 8),
            (WaveBuffer::ReplaySpanExtents, 6 * 2 * 4),
        ] {
            assert_eq!(
                b.bytes(&g, replay),
                want,
                "{b:?} against its measured carve"
            );
        }
        // A replay prices *only* that chain: no attention, no FFN, no head.
        assert_eq!(
            plan.phase_bytes(LayerPhase::Attention, replay),
            plan.chain_bytes(Chain::DeltaNetReplay, replay)
        );
        assert_eq!(plan.phase_bytes(LayerPhase::Ffn, replay), 0);
        assert_eq!(plan.phase_bytes(LayerPhase::Forward, replay), 0);
        // And an ordinary wave is charged nothing for it, however wide.
        let ordinary = WaveWidth::prefill(8192, 64);
        assert_eq!(plan.chain_bytes(Chain::DeltaNetReplay, ordinary), 0);
    }

    /// **A wave that stops short of the last layer runs no head**, so it
    /// reserves none of the forward phase. A segmented sweep prices one window
    /// per layer range, and charging the head's full width to each was 17.4 MiB
    /// a window on a 60-session Qwen2.
    #[test]
    fn a_wave_that_runs_no_head_prices_no_forward_phase() {
        let plan = WavePlan::new(gated_partial_rotary());
        let no_head = WaveWidth {
            prefill_rows: 2100,
            ..WaveWidth::default()
        };
        assert_eq!(plan.phase_bytes(LayerPhase::Forward, no_head), 0);
        // The layer phases are untouched — the window still runs its layers.
        assert!(plan.phase_bytes(LayerPhase::Attention, no_head) > 0);
        assert!(plan.phase_bytes(LayerPhase::Ffn, no_head) > 0);
    }

    /// **A hybrid's two mixers are a max, never a sum.** A layer runs one of
    /// them, so charging both prices a wave for a layer that does not exist —
    /// and charging neither under-sizes the arena the other one carves from.
    #[test]
    fn the_attention_phase_takes_the_larger_mixer_chain() {
        let g = gated_partial_rotary();
        let plan = WavePlan::new(g);
        let rows = WaveWidth::prefill(2100, 4);
        let attn = plan.chain_bytes(Chain::Attention, rows);
        let dn = plan.chain_bytes(Chain::DeltaNet, rows);
        assert!(dn > 0, "a hybrid geometry must price its mixer chain");
        assert_eq!(
            plan.phase_bytes(LayerPhase::Attention, rows),
            attn.max(dn),
            "the phase must be the larger chain, not their sum ({attn} + {dn})"
        );
        // And an all-attention stack prices its mixer chain at nothing at all,
        // alignment included, so no such model's span moves.
        let plain = WavePlan::new(moe());
        assert_eq!(plain.chain_bytes(Chain::DeltaNet, rows), 0);
        assert_eq!(
            plain.phase_bytes(LayerPhase::Attention, rows),
            plain.chain_bytes(Chain::Attention, rows)
        );
    }

    /// Per-row cost of the attention chain a **prefill** group runs, as
    /// a `wave-census` build measured it on Qwen3-30B-A3B: twelve carves, and every
    /// one of them a whole number of bytes per row.
    ///
    /// ```text
    /// [ 0]    2304  AttnNorm       q8a128 over hidden
    /// [ 1]   10240  QkvProjection  the fused segmented launch
    /// [ 2]    8192  QSplit         narrow out of the fused buffer
    /// [ 3]    8192  QNormIn        flatten for the head-wise norm
    /// [ 4]    8192  QNormOut       the norm itself
    /// [ 5]    8192  QHeadsPacked   transpose back
    /// [ 6]    1024  KSplit
    /// [ 7]    1024  KNormIn
    /// [ 8]    1024  KNormOut
    /// [ 9]    1024  KHeadsPacked
    /// [10]    1024  VContiguous
    /// [11]    8192  AttnOutput     the paged prefill kernel's context
    /// ```
    const MEASURED_ATTN_PREFILL_PER_ROW: usize = 58624;

    /// The same for a **decode** group: nine carves, and a different set — the
    /// `seq == 1` reshapes are free, and the context comes back already packed,
    /// so `o_proj` runs off it and lands on the span.
    ///
    /// ```text
    /// [0]  2304  AttnNorm
    /// [1] 10240  QkvProjection
    /// [2]  8192  QSplit
    /// [3]  8192  QNormOut
    /// [4]  1024  KSplit
    /// [5]  1024  KNormOut
    /// [6]  1024  VContiguous
    /// [7]  4608  DecodeContext   q8a1024, emitted by the decode kernel
    /// [8]  4096  OProjOutput
    /// ```
    const MEASURED_ATTN_DECODE_PER_ROW: usize = 40704;

    /// The same decode chain at **one row**, where `QSplit`, `KSplit` and
    /// `VContiguous` are not carved: the narrows out of a single fused row are
    /// already contiguous. Read off the per-forward pair on the Qwen3-30B-A3B
    /// gate — planned 40,704 B, used 30,464 B at `0 prefill + 1 decode rows`.
    const MEASURED_ATTN_DECODE_ONE_ROW: usize = 30464;

    /// The FFN chain per row on a **BF16** int8 session, excluding the routing
    /// tables (which do not scale by the row alone):
    ///
    /// ```text
    ///  2304  FfnNorm        q8a128 over hidden
    ///   256  RouterLogits
    ///    64  RouteWeights + RouteIndices
    ///  4096  MoeCombine
    /// 18432  MoeGather      8 × q8a128 over hidden
    /// 49152  Gate + Up      8 × 768 × F32, twice
    ///  6912  SwigluAct      8 × q8a128 over 768
    /// 65536  DownGemm       8 × 2048 × F32
    /// ```
    const MEASURED_FFN_PIPELINE_PER_ROW: usize = 146752;

    /// Two whole FFN generations off the census, to the byte — the plan must
    /// price each exactly, routing tables and alignment pad included.
    ///
    /// * BF16, 4,960 rows: 14 carves, no result cast, 124 B lost to alignment.
    /// * F16, 9,880 rows: 15 carves, the `to_dtype_mut` result cast among them,
    ///   156 B lost to alignment.
    const MEASURED_FFN_BF16_4960: usize = 728_386_048;
    const MEASURED_FFN_F16_9880: usize = 1_491_366_400;

    /// The same generation on Qwen3.5-35B-A3B, whose MoE adds a shared expert:
    /// six more carves (gate/up, SiLU, product, gate tile, sigmoid, the final
    /// sum) and the alignment pads the sigmoid and the routing tables leave.
    ///
    /// * BF16, 2,100 rows: 20 carves, 836 B lost to alignment.
    /// * F16, 1,070 rows: 21 carves — the result cast again — 680 B lost.
    const MEASURED_FFN_35B_BF16_2100: usize = 287_024_640;
    const MEASURED_FFN_35B_F16_1070: usize = 150_628_864;

    /// Qwen3.5-35B-A3B's FFN shapes. The FFN chain takes nothing from the
    /// attention side, so only the fields it reads are the checkpoint's.
    fn moe_35b() -> ModelGeometry {
        ModelGeometry {
            intermediate: 512,
            experts_per_tok: 8,
            n_experts: 256,
            shared_expert: Some(SharedExpertWidths {
                intermediate: 512,
                gate_cols: 32,
            }),
            ..moe()
        }
    }

    /// The plan must cover every chain that can run in its phase, because a
    /// mixed wave runs more than one of them inside a single generation.
    ///
    /// This is the assertion the whole module exists to make true, and it was
    /// false by 1.8x on attention and by 2x on the accumulate dtype until the
    /// **A one-row wave is priced without the three fused-row copies**, because
    /// it does not make them — and only a one-row wave: at two rows every
    /// narrow strides over the row dimension and each copy is real.
    #[test]
    fn a_single_row_wave_copies_nothing_out_of_the_fused_row() {
        let g = moe();
        let plan = WavePlan::new(g);
        assert_eq!(
            plan.phase_bytes(LayerPhase::Attention, WaveWidth::decode(1)),
            MEASURED_ATTN_DECODE_ONE_ROW,
            "{}",
            plan.describe(WaveWidth::decode(1))
        );
        for b in [
            WaveBuffer::QSplit,
            WaveBuffer::KSplit,
            WaveBuffer::VContiguous,
        ] {
            assert_eq!(b.bytes(&g, WaveWidth::decode(1)), 0, "{b:?} at one row");
            assert_eq!(b.bytes(&g, WaveWidth::prefill(1, 1)), 0, "{b:?} at one row");
            assert!(b.bytes(&g, WaveWidth::decode(2)) > 0, "{b:?} at two rows");
        }
        // A gated lineage's Q is interleaved per head, so its narrow strides
        // across heads even at one row.
        let gated = ModelGeometry {
            gated_qkv: true,
            ..g
        };
        assert!(WaveBuffer::QSplit.bytes(&gated, WaveWidth::decode(1)) > 0);
    }

    /// **With Q/K/V biases the adds are the three buffers, at every width.**
    /// The add reads the narrowed view as it stands, so no split copy happens —
    /// and unlike the copy, the add is not free at one row. Priced through the
    /// split copies, Qwen2 matched to the byte until a one-row decode, where it
    /// ran out of span 1,792 B (one Q row) into the Q bias.
    #[test]
    fn a_bias_add_replaces_the_split_copy_at_every_width() {
        let biased = ModelGeometry {
            qkv_bias: true,
            ..moe()
        };
        for width in [
            WaveWidth::decode(1),
            WaveWidth::prefill(1, 1),
            WaveWidth::decode(7),
        ] {
            let rows = width.rows();
            assert_eq!(
                WaveBuffer::QBias.bytes(&biased, width),
                rows * biased.attn_cols() * 2
            );
            for b in [WaveBuffer::KBias, WaveBuffer::VBias] {
                assert_eq!(b.bytes(&biased, width), rows * biased.kv_cols() * 2);
            }
            for b in [
                WaveBuffer::QSplit,
                WaveBuffer::KSplit,
                WaveBuffer::VContiguous,
            ] {
                assert_eq!(b.bytes(&biased, width), 0, "{b:?}");
            }
        }
        // Qwen2-0.5B's own decode generation, off the gate: 60 decode rows,
        // head dim 64 — so the FP context, not the q8 one — and the three bias
        // adds. The norm, the fused projection, the adds and one `rows ×
        // hidden` buffer after them, plus the pad the flat-grouped q8 norm
        // leaves.
        let qwen2 = ModelGeometry {
            hidden: 896,
            vocab: 151_936,
            intermediate: 4864,
            n_head: 14,
            n_kv_head: 2,
            head_dim: 64,
            experts_per_tok: 1,
            n_experts: 1,
            head_qk_norm: false,
            packed_head: false,
            decode_q8_context: false,
            ..biased
        };
        assert_eq!(
            WavePlan::new(qwen2).phase_bytes(LayerPhase::Attention, WaveWidth::decode(60)),
            445_184,
            "{}",
            WavePlan::new(qwen2).describe(WaveWidth::decode(60))
        );
        // Past one row the phase costs what the split copies used to, which is
        // why the old pricing matched every wide wave.
        let width = WaveWidth::decode(7);
        assert_eq!(
            WavePlan::new(biased).phase_bytes(LayerPhase::Attention, width),
            WavePlan::new(moe()).phase_bytes(LayerPhase::Attention, width),
        );
    }

    #[test]
    fn the_plan_covers_every_measured_chain() {
        let plan = WavePlan::new(moe());
        // From two rows: a one-row wave carves a different chain, pinned by
        // `a_single_row_wave_copies_nothing_out_of_the_fused_row`.
        for rows in [2usize, 20, 124, 744, 3936] {
            // **Each group against its own width**, which is what the split
            // buys: a wave of `rows` prefill tokens must cover the prefill
            // chain at `rows`, and a wave of `rows` decode tokens the decode
            // chain at `rows` — but neither is asked to cover the other's.
            for (name, rate, width) in [
                (
                    "prefill",
                    MEASURED_ATTN_PREFILL_PER_ROW,
                    WaveWidth::prefill(rows, 1),
                ),
                (
                    "decode",
                    MEASURED_ATTN_DECODE_PER_ROW,
                    WaveWidth::decode(rows),
                ),
            ] {
                let attn = plan.phase_bytes(LayerPhase::Attention, width);
                assert!(
                    attn >= rate * rows,
                    "attention at {rows} {name} rows prices {attn} B but the \
                     measured chain takes {} B\n{}",
                    rate * rows,
                    plan.describe(width)
                );
            }
            let width = WaveWidth::prefill(rows, 1);
            let ffn = plan.phase_bytes(LayerPhase::Ffn, width);
            assert!(
                ffn >= MEASURED_FFN_PIPELINE_PER_ROW * rows,
                "FFN at {rows} rows prices {ffn} B but the measured pipeline \
                 chain takes {} B\n{}",
                MEASURED_FFN_PIPELINE_PER_ROW * rows,
                plan.describe(width)
            );
        }
    }

    /// What the union costs over the widest single chain, recorded so the
    /// over-bound is a number someone chose rather than one nobody noticed.
    ///
    /// **A pure-prefill wave pays nothing for the decode chain, and vice
    /// versa** — the margin this used to record is gone, and its absence is the
    /// assertion.
    ///
    /// It was 17.9%, then 14.8% once the norm was priced in the session's real
    /// encoding ([`ModelGeometry::packed_norm`]), and it is zero now that each
    /// buffer is sized by its own group's rows rather than by the wave's total.
    /// The doc here used to say "closing it means passing the split"; this is
    /// what that looked like.
    ///
    /// A **mixed** wave still pays for both, and must — one attention
    /// generation holds the decode group's buffers and the prefill group's at
    /// once — but each at its own width, which is a sum of two measurements
    /// rather than a bound over the larger.
    #[test]
    fn each_group_pays_for_its_own_chain_and_not_the_others() {
        let plan = WavePlan::new(moe());
        let rows = 1000;

        let prefill_only = plan.phase_bytes(LayerPhase::Attention, WaveWidth::prefill(rows, 1));
        assert_eq!(
            prefill_only,
            MEASURED_ATTN_PREFILL_PER_ROW * rows,
            "a wave with no decode rows must price the prefill chain exactly"
        );

        let decode_only = plan.phase_bytes(LayerPhase::Attention, WaveWidth::decode(rows));
        assert_eq!(
            decode_only,
            MEASURED_ATTN_DECODE_PER_ROW * rows,
            "a wave with no prefill rows must price the decode chain exactly"
        );

        // Mixed: both chains live in one generation, each at its own width.
        let mixed = plan.phase_bytes(
            LayerPhase::Attention,
            WaveWidth {
                prefill_rows: rows,
                decode_rows: rows,
                scored_rows: rows + 1,
                ..WaveWidth::default()
            },
        );
        assert_eq!(mixed, prefill_only + decode_only);
    }

    #[test]
    fn the_ffn_phase_dominates_a_moe_layer() {
        let plan = WavePlan::new(moe());
        assert!(
            plan.phase_bytes(LayerPhase::Ffn, w(64))
                > plan.phase_bytes(LayerPhase::Attention, w(64)),
            "expert replication should make the FFN the sizing phase:\n{}",
            plan.describe(w(64))
        );
    }

    /// A wave costs the larger phase, never the sum: the two generations do not
    /// overlap, so summing would price twice what a half holds at once.
    #[test]
    fn a_wave_costs_the_larger_phase_not_the_sum() {
        let plan = WavePlan::new(moe());
        let sum = plan.phase_bytes(LayerPhase::Attention, w(64))
            + plan.phase_bytes(LayerPhase::Ffn, w(64));
        assert!(plan.wave_bytes(w(64)) < sum);
        assert!(plan.wave_bytes(w(64)) >= plan.phase_bytes(LayerPhase::Ffn, w(64)));
    }

    /// The bisection in `max_rows_within` is only valid if cost never decreases
    /// with width. Assert the property it relies on rather than trusting it.
    #[test]
    fn wave_cost_is_non_decreasing_in_rows() {
        for g in [moe(), dense()] {
            let plan = WavePlan::new(g);
            let mut prev = 0;
            for rows in 1..600 {
                let cost = plan.wave_bytes(w(rows));
                assert!(cost >= prev, "cost fell from {prev} at {rows} rows");
                prev = cost;
            }
        }
    }

    /// The admission bound must be exact: the returned width fits and one more
    /// row does not. An off-by-one here over-admits, and the span refuses
    /// mid-forward instead of at the gate.
    #[test]
    fn max_rows_within_is_the_exact_boundary() {
        for g in [moe(), dense()] {
            let plan = WavePlan::new(g);
            for budget in [4 * MIB, 16 * MIB, 64 * MIB, 256 * MIB] {
                let rows = plan.max_rows_within(budget, empty_head());
                assert!(rows > 0, "budget {budget} should fit at least one row");
                assert!(
                    plan.fits(w(rows), budget),
                    "{rows} rows must fit in {budget} B"
                );
                assert!(
                    !plan.fits(w(rows + 1), budget),
                    "{} rows must NOT fit in {budget} B",
                    rows + 1
                );
            }
        }
    }

    /// The gate must accept exactly what `max_rows_within` promised and refuse
    /// the next row — an `ensure_fits` looser than the bound would let a wave
    /// through that the span then refuses mid-forward, which is the failure the
    /// pre-flight exists to move earlier.
    #[test]
    fn ensure_fits_agrees_with_the_admission_bound() {
        let plan = WavePlan::new(moe());
        let budget = 32 * MIB;
        let widest = plan.max_rows_within(budget, empty_head());
        assert!(plan.ensure_fits(w(widest), budget).is_ok());
        assert!(plan.ensure_fits(w(widest + 1), budget).is_err());
    }

    /// An over-budget refusal has to say enough to act on: how wide the wave
    /// was, how much it overran, and what would have fit.
    #[test]
    fn the_over_budget_error_names_the_width_and_the_overage() {
        let plan = WavePlan::new(moe());
        let budget = 8 * MIB;
        let rows = plan.max_rows_within(budget, empty_head()) + 64;
        let err = plan
            .ensure_fits(w(rows), budget)
            .expect_err("must refuse")
            .to_string();
        assert!(err.contains("wave over budget"), "{err}");
        assert!(err.contains(&rows.to_string()), "names the width: {err}");
        assert!(err.contains("over by"), "names the overage: {err}");
        assert!(
            err.contains("widest prefill"),
            "names what would fit: {err}"
        );
        assert!(err.contains("scored"), "names the scored rows: {err}");
    }

    /// A budget too small to price a single token is a misconfiguration, and
    /// must be reported as zero rather than rounded up to one — admitting a
    /// wave the span cannot hold is the failure the gate exists to prevent.
    ///
    /// At exactly one row's price it admits the widest width that price still
    /// covers — which need not be one. `wave_bytes` is the largest phase, and
    /// at one prefill row that is the head's single scored row of logits, which
    /// a second prefill row does not add to: the FFN at two rows still sits
    /// under it. The staircase has a flat step there, and the bound is only
    /// right if it walks to the end of it.
    #[test]
    fn a_budget_below_one_row_admits_nothing() {
        let plan = WavePlan::new(moe());
        let one_row = plan.wave_bytes(w(1));
        assert_eq!(plan.max_rows_within(one_row - 1, empty_head()), 0);
        let widest = plan.max_rows_within(one_row, empty_head());
        assert!(widest >= 1);
        assert!(plan.wave_bytes(w(widest)) <= one_row);
        assert!(plan.wave_bytes(w(widest + 1)) > one_row);
    }

    /// Halving the budget must roughly halve the admitted width — the property
    /// that makes the half size a usable throughput/footprint dial.
    #[test]
    fn the_admitted_width_tracks_the_budget() {
        let plan = WavePlan::new(moe());
        let wide = plan.max_rows_within(64 * MIB, empty_head());
        let narrow = plan.max_rows_within(32 * MIB, empty_head());
        let ratio = wide as f64 / narrow as f64;
        assert!(
            (1.8..=2.2).contains(&ratio),
            "half the budget should admit about half the rows, got {ratio:.2}x"
        );
    }

    /// A MoE model replicates every token across its routed experts, so at the
    /// same budget it must admit fewer rows than the same shape routed to one.
    #[test]
    fn expert_replication_narrows_the_admitted_wave() {
        let routed = WavePlan::new(moe());
        let single = WavePlan::new(ModelGeometry {
            experts_per_tok: 1,
            ..moe()
        });
        assert!(
            single.max_rows_within(64 * MIB, empty_head())
                > routed.max_rows_within(64 * MIB, empty_head()),
            "routing to 8 experts must narrow the wave"
        );
    }

    /// A wider accumulate dtype is real cost — a plan ignoring it would
    /// over-admit on every int8 layer.
    #[test]
    fn the_accumulate_dtype_narrows_the_admitted_wave() {
        let f32_accum = WavePlan::new(moe());
        let bf16_accum = WavePlan::new(ModelGeometry {
            accum_dtype: DType::BF16,
            ..moe()
        });
        assert!(
            bf16_accum.max_rows_within(64 * MIB, empty_head())
                > f32_accum.max_rows_within(64 * MIB, empty_head())
        );
    }

    /// What the FFN span actually admits on the production model, recorded so a
    /// change to either the span or the geometry has to face the number.
    ///
    /// The compute-side ceiling (`MAX_PREFILL_TOKENS`) and this are independent,
    /// and the caller takes the narrower. Printing both is what tells you which
    /// one is binding — a question that has no answer from either constant alone.
    #[test]
    fn the_ffn_span_admits_a_recorded_width() {
        use super::super::wave_spans::WAVE_FFN_BYTES;
        let plan = WavePlan::new(moe());
        let rows = plan.max_rows_within(WAVE_FFN_BYTES, empty_head());
        println!(
            "FFN span {} B admits {rows} rows; one row costs {} B",
            WAVE_FFN_BYTES,
            plan.phase_bytes(LayerPhase::Ffn, w(1)),
        );
        assert!(
            rows > 0,
            "a span that cannot price one row would stall every wave"
        );
        // Monotonic in the budget: halving the span must not admit more rows.
        assert!(plan.max_rows_within(WAVE_FFN_BYTES / 2, empty_head()) <= rows);
    }
}
