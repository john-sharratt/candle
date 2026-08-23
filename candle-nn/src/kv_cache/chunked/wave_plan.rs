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
//! attention or FFN chain is a change to this list**, and `KV_WAVE_CENSUS=1`
//! over the gate is how to find out what it should say.
//!
//! # A phase at a time, and the union of the chains that can run in it
//!
//! A layer opens two generations — one spanning attention → `o_proj`, one
//! spanning the FFN — and each drops before the next opens, resetting its span.
//! So a span holds **one layer phase**, never a whole wave, and each phase gets
//! its own arena ([`super::bump_arena`]) sized from its own peak rather than
//! both from the larger.
//!
//! Within a phase more than one chain can run, and the plan charges their
//! **union** rather than their maximum. That is not conservatism, it is the
//! shape of a mixed wave: `forward_layer_batched_mixed` opens **one** attention
//! generation and runs every group inside it, so a wave carrying both a decode
//! group and a prefill group allocates the decode chain's buffers *and* the
//! prefill chain's, into the same span, before either guard drops. The prefill
//! chain is the wider of the two per row, so a wave that is entirely prefill
//! pays for the decode chain's two extra buffers it never allocates — about 15%
//! of the attention phase, which is the price of the plan being given a total
//! row count rather than a per-group split.
//!
//! The same holds for the FFN's two expert-dispatch paths, except that the
//! GPU-native path's buffers are a subset of the pipeline path's, so the union
//! costs nothing there.

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
    /// Carries no [`WaveBuffer`] variants, so it prices as zero. That is
    /// deliberate — the plan sizes what scales with wave *width*, and this holds
    /// a few kilobytes of per-forward metadata whose size is set by the sequence
    /// count, not by the token count.
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
    /// The compute dtype activations are carried in.
    pub act_dtype: DType,
    /// What the int8 tensor-core kernels emit before the cast back to
    /// `act_dtype`. Both are live at once, so both are planned.
    pub accum_dtype: DType,
    /// Whether the Q/K/V projections round-trip through [`Self::accum_dtype`].
    ///
    /// A packed session's projections consume the norm's q8a128 output and the
    /// tensor-core epilogue emits `act_dtype`, so the projection costs what
    /// [`WaveBuffer::QkvProjection`] says and nothing more. A **float** session
    /// runs the dequantized GEMM in `accum_dtype` instead: it upcasts its
    /// operand, computes there, and casts the result back down — three extra
    /// full-width buffers per projection, none of which the packed path
    /// allocates.
    ///
    /// A flag rather than an unconditional charge because the two are
    /// alternatives fixed at session creation, and charging both would price
    /// every packed model for a round trip it never runs — on Qwen3-30B-A3B
    /// that is a 95% over-bound on the attention span, against a chain whose
    /// census shows no upcast at all.
    pub projection_accum_roundtrip: bool,
    /// The Q projection emits `2 × head_dim` per head — interleaved
    /// `[q | gate]` — and the gate travels the whole attention block: split
    /// out contiguously, sigmoided, and multiplied into the context. Widens
    /// `qkv_cols` and charges the three gate-side buffers. Gated lineages
    /// (Qwen3.5/3.8) set it; classic attention leaves it false and pays
    /// nothing.
    pub gated_qkv: bool,
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
}

/// How many u32 of routing metadata the expert pipeline uploads per assignment.
///
/// The one term here that is a **bound rather than a transcript**, and it is
/// labelled as such because the alternative is worse. The threaded expert
/// pipeline splits a wave's assignments into expert batches and, per batch,
/// uploads three assignment-indexed tables, three tile-indexed tables (a tile
/// covers at least one assignment, so these are bounded by the same count) and
/// one token-indexed table. Batches partition the assignments, so however many
/// there are the assignment-indexed uploads sum to the same total — but the
/// tile and token tables do not decompose that cleanly, and pinning them exactly
/// would tie this file to the pipeline's batching policy.
///
/// The census measured 3.5 u32 per assignment at the widest wave the gate runs.
/// Eight is that doubled, and it costs 0.19% of the FFN phase — a margin worth
/// paying on the plan's least structured term, where being short fails a forward
/// and being generous is invisible.
const ROUTING_U32_PER_ASSIGNMENT: usize = 8;

/// Every buffer a layer allocates from the transient tier.
///
/// One variant per allocation site, and the list is the **union** over the
/// chains a phase can run — see the module header for why a union rather than a
/// maximum. Every one of these was read off [`super::wave_census`] on
/// Qwen3-30B-A3B; the comment on each says which chain allocates it, so a
/// reader can check the list against the code that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum WaveBuffer {
    /// Attention RMSNorm, in whatever encoding the QKV matmul consumes. Both
    /// chains.
    ///
    /// **Priced dense even though an int8 session's norm emits q8a128.** The
    /// two encodings are alternatives — int8 mode is fixed when the session is
    /// created — so a phase holds one or the other, and dense is the larger by
    /// about 1.8×. Pricing the larger keeps the plan an upper bound for either
    /// mode, which is what [`WavePlan::ensure_fits`] needs; pricing the smaller
    /// would under-size a float session's span, and the buffer squeezed out at
    /// the end of the phase is one a kernel wrapper allocates, which *refuses*
    /// rather than falling back to the pool.
    AttnNorm,
    /// Fused Q/K/V projection output — one segmented launch over the three KO
    /// weights, narrowed into q/k/v afterwards. Both chains.
    QkvProjection,
    /// The projection's operand, upcast to the GEMM's accumulate dtype.
    ///
    /// A float-path projection runs in `accum_dtype`: the norm's `act_dtype`
    /// output is upcast, the GEMM emits in `accum_dtype`, and the result is
    /// cast back down. Both ends stay live for the whole phase — the bump
    /// cursor does not rewind — so both are priced, the same round trip
    /// [`Self::GateGemm`] / [`Self::UpGemm`] / [`Self::DownGemm`] are priced
    /// for on the FFN side.
    ///
    /// Charged for **three** `hidden`-wide operands, which is the upper bound
    /// over both chains: a model that projects Q, K and V separately upcasts
    /// the norm output once per projection, and a fused projection upcasts it
    /// once.
    QkvProjOperand,
    /// The projection's result in `accum_dtype`, before the cast to
    /// `act_dtype`.
    ///
    /// `qkv_cols` wide, which again bounds both chains: a fused projection
    /// emits exactly that, and separate Q/K/V results sum to it.
    QkvProjAccum,
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
    /// `o_proj`'s operand upcast to `accum_dtype` — the output projection's
    /// half of the float-session round trip
    /// ([`ModelGeometry::projection_accum_roundtrip`]), which the plan models
    /// for Q/K/V and must model for O the same way.
    OProjOperand,
    /// `o_proj`'s result in `accum_dtype`, before the cast back to
    /// `act_dtype`. Same condition as [`Self::OProjOperand`].
    OProjAccum,
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
    /// On an **int8** session prefill's `o_proj` takes a `Float` context, the
    /// override quantizes it at the matmul, and that quantize breaks the
    /// provenance chain — the output lands on the pool, and only decode
    /// carves here. On a **float** session (`projection_accum_roundtrip`)
    /// prefill's `o_proj` result rides the span like everything else, so the
    /// charge covers both.
    OProjOutput,
    /// FFN RMSNorm, in whatever encoding the expert GEMMs consume. Both
    /// dispatch paths, and priced dense for the reason given on
    /// [`Self::AttnNorm`].
    FfnNorm,
    /// The router's per-token logits over every expert. Both paths.
    RouterLogits,
    /// Top-k routing weights, in F32. Both paths.
    RouteWeights,
    /// Top-k expert ids, as u32. Both paths.
    RouteIndices,
    /// The routing tables the expert pipeline uploads per batch — see
    /// [`ROUTING_U32_PER_ASSIGNMENT`]. Threaded pipeline only.
    RoutingTables,
    /// Tokens gathered into expert-major order for the grouped GEMMs.
    MoeGather,
    /// Gate projection over the gathered tokens.
    GateGemm,
    /// Up projection over the gathered tokens.
    UpGemm,
    /// Fused SwiGLU output, requantized to feed the down GEMM.
    SwigluAct,
    /// Down projection, in the accumulate dtype.
    ///
    /// Threaded pipeline only: the GPU-native path's grouped down GEMM feeds
    /// `fused_deterministic_scatter` without materialising a span-backed result.
    DownGemm,
    /// Down projection cast to the compute dtype for the scatter. Pipeline only.
    DownCast,
    /// The MoE combine target the scatter accumulates into. Both paths.
    MoeCombine,
    /// The FFN result cast to the residual's dtype before the second residual
    /// add.
    ///
    /// A no-op when the two already agree, which is every BF16 configuration —
    /// but an F16 session carries its residual in F16 and its MoE in BF16, and
    /// then this is a full `rows × hidden` buffer at the very end of the phase,
    /// when the span is at its fullest.
    FfnResidualCast,
}

impl WaveBuffer {
    /// When this buffer is live within its phase.
    pub fn phase(&self) -> LayerPhase {
        match self {
            Self::AttnNorm
            | Self::QkvProjection
            | Self::QkvProjOperand
            | Self::QkvProjAccum
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
            | Self::AttnOutput
            | Self::DecodeContext
            | Self::OProjOperand
            | Self::OProjAccum
            | Self::OProjOutput => LayerPhase::Attention,
            Self::FfnNorm
            | Self::RouterLogits
            | Self::RouteWeights
            | Self::RouteIndices
            | Self::RoutingTables
            | Self::MoeGather
            | Self::GateGemm
            | Self::UpGemm
            | Self::SwigluAct
            | Self::DownGemm
            | Self::DownCast
            | Self::MoeCombine
            | Self::FfnResidualCast => LayerPhase::Ffn,
        }
    }

    /// This buffer's shape for a wave of `rows` tokens — the declaration the
    /// call site and the admission gate both size from.
    ///
    /// Also exhaustive: a new variant must state its shape here, and takes its
    /// byte count from [`BufferShape::bytes`] rather than writing arithmetic of
    /// its own.
    pub fn shape(&self, g: &ModelGeometry, rows: usize) -> BufferShape {
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
        let er = g.expert_rows(rows);
        match self {
            Self::AttnNorm => dense(rows, g.hidden, g.act_dtype),
            Self::QkvProjection => dense(rows, g.qkv_cols(), g.act_dtype),
            Self::QkvProjOperand if g.projection_accum_roundtrip => {
                dense(rows, 3 * g.hidden, g.accum_dtype)
            }
            Self::QkvProjAccum if g.projection_accum_roundtrip => {
                dense(rows, g.qkv_cols(), g.accum_dtype)
            }
            Self::QkvProjOperand | Self::QkvProjAccum => dense(0, 0, g.accum_dtype),
            Self::QSplit | Self::QNormIn | Self::QNormOut | Self::QHeadsPacked => {
                dense(rows, g.attn_cols(), g.act_dtype)
            }
            Self::GateSplit | Self::GateSigmoid | Self::GatedContext if g.gated_qkv => {
                dense(rows, g.attn_cols(), g.act_dtype)
            }
            Self::GateSplit | Self::GateSigmoid | Self::GatedContext => dense(0, 0, g.act_dtype),
            Self::QRotaryPermute if g.partial_rotary => dense(rows, g.attn_cols(), g.act_dtype),
            Self::KRotaryPermute if g.partial_rotary => dense(rows, g.kv_cols(), g.act_dtype),
            Self::QRotaryPermute | Self::KRotaryPermute => dense(0, 0, g.act_dtype),
            Self::KSplit
            | Self::KNormIn
            | Self::KNormOut
            | Self::KHeadsPacked
            | Self::VContiguous => dense(rows, g.kv_cols(), g.act_dtype),
            Self::AttnOutput => dense(rows, g.attn_cols(), g.act_dtype),
            Self::DecodeContext => q8(rows, g.attn_cols()),
            Self::OProjOperand if g.projection_accum_roundtrip => {
                dense(rows, g.attn_cols(), g.accum_dtype)
            }
            Self::OProjAccum if g.projection_accum_roundtrip => {
                dense(rows, g.hidden, g.accum_dtype)
            }
            Self::OProjOperand | Self::OProjAccum => dense(0, 0, g.accum_dtype),
            Self::OProjOutput => dense(rows, g.hidden, g.act_dtype),
            Self::FfnNorm => dense(rows, g.hidden, g.act_dtype),
            Self::RouterLogits => dense(rows, g.n_experts, g.act_dtype),
            Self::RouteWeights => dense(rows, g.experts_per_tok, DType::F32),
            Self::RouteIndices => dense(rows, g.experts_per_tok, DType::U32),
            Self::RoutingTables => dense(er, ROUTING_U32_PER_ASSIGNMENT, DType::U32),
            Self::MoeGather => q8(er, g.hidden),
            Self::GateGemm => dense(er, g.intermediate, g.accum_dtype),
            Self::UpGemm => dense(er, g.intermediate, g.accum_dtype),
            Self::SwigluAct => q8(er, g.intermediate),
            Self::DownGemm => dense(er, g.hidden, g.accum_dtype),
            Self::DownCast => dense(er, g.hidden, g.act_dtype),
            Self::MoeCombine => dense(rows, g.hidden, g.act_dtype),
            Self::FfnResidualCast => dense(rows, g.hidden, g.act_dtype),
        }
    }

    /// Bytes this buffer needs for a wave of `rows` tokens.
    pub fn bytes(&self, g: &ModelGeometry, rows: usize) -> usize {
        self.shape(g, rows).bytes()
    }
}

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
    pub fn bytes(&self, buffer: WaveBuffer, rows: usize) -> usize {
        buffer.bytes(&self.geometry, rows)
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
    pub fn phase_bytes(&self, phase: LayerPhase, rows: usize) -> usize {
        WaveBuffer::iter()
            .filter(|b| b.phase() == phase)
            .map(|b| b.bytes(&self.geometry, rows) + BUMP_ALIGNMENT)
            .sum()
    }

    pub fn wave_bytes(&self, rows: usize) -> usize {
        LayerPhase::iter()
            .map(|p| self.phase_bytes(p, rows))
            .max()
            .unwrap_or(0)
    }

    /// The widest wave that fits in `budget` bytes — the admission bound.
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
    pub fn max_rows_within(&self, budget: usize) -> usize {
        if self.wave_bytes(1) > budget {
            return 0;
        }
        let mut lo = 1usize;
        let mut hi = 2usize;
        while self.wave_bytes(hi) <= budget {
            lo = hi;
            hi = hi.saturating_mul(2);
        }
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if self.wave_bytes(mid) <= budget {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Whether a wave of `rows` fits in `budget`.
    pub fn fits(&self, rows: usize, budget: usize) -> bool {
        self.wave_bytes(rows) <= budget
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
    pub fn ensure_fits(&self, rows: usize, budget: usize) -> candle::Result<()> {
        let cost = self.wave_bytes(rows);
        if cost <= budget {
            return Ok(());
        }
        let widest = self.max_rows_within(budget);
        candle::bail!(
            "wave over budget: {rows} rows need {cost} B of transient span but the \
             half holds {budget} B (over by {} B). The widest wave this budget \
             admits is {widest} rows. Admission priced this wave against the same \
             plan, so this is an accounting disagreement, not a wave to trim.\n{}",
            cost - budget,
            self.describe(rows)
        )
    }

    /// One line per phase and one per buffer within it — what to print when a
    /// span refuses, so an overflow names a shape rather than a number.
    pub fn describe(&self, rows: usize) -> String {
        let mut out = format!("wave plan @ {rows} rows: {} B\n", self.wave_bytes(rows));
        for phase in LayerPhase::iter() {
            out.push_str(&format!(
                "  {phase:?}: {} B\n",
                self.phase_bytes(phase, rows)
            ));
            for buffer in WaveBuffer::iter().filter(|b| b.phase() == phase) {
                out.push_str(&format!(
                    "    {buffer:?}: {} B (+{BUMP_ALIGNMENT} B alignment)\n",
                    buffer.bytes(&self.geometry, rows)
                ));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3-30B-A3B's real shapes.
    fn moe() -> ModelGeometry {
        ModelGeometry {
            hidden: 2048,
            intermediate: 768,
            n_head: 32,
            n_kv_head: 4,
            head_dim: 128,
            experts_per_tok: 8,
            n_experts: 128,
            act_dtype: DType::BF16,
            accum_dtype: DType::F32,
            // Its census shows packed projections — no upcast, no cast back.
            projection_accum_roundtrip: false,
            gated_qkv: false,
            partial_rotary: false,
        }
    }

    /// A float-projection stack: Qwen3.5-0.8B's real shapes, whose dequantized
    /// Q/K/V GEMMs run in F32 and round-trip through it. Gated `[q | gate]`
    /// projection and 64-of-256 partial rotary. The head count is 8, read off
    /// the census (the q matmul emits `2 · 8 · 256` accum columns; an earlier
    /// fixture said 16 ungated — which priced the same `qkv_cols` by accident
    /// and hid the gate's whole downstream chain).
    fn float_projection() -> ModelGeometry {
        ModelGeometry {
            hidden: 1024,
            intermediate: 3072,
            n_head: 8,
            n_kv_head: 2,
            head_dim: 256,
            experts_per_tok: 1,
            n_experts: 1,
            act_dtype: DType::BF16,
            accum_dtype: DType::F32,
            projection_accum_roundtrip: true,
            gated_qkv: true,
            partial_rotary: true,
        }
    }

    fn dense() -> ModelGeometry {
        ModelGeometry {
            hidden: 4096,
            intermediate: 12288,
            n_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            experts_per_tok: 1,
            n_experts: 1,
            act_dtype: DType::BF16,
            accum_dtype: DType::F32,
            projection_accum_roundtrip: false,
            gated_qkv: false,
            partial_rotary: false,
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
        for g in [float_projection(), moe(), dense()] {
            for rows in [1usize, 64, 4096] {
                for b in WaveBuffer::iter() {
                    // The projection round-trip pair is the one conditional
                    // shape in the list: it prices zero exactly when the
                    // geometry says the chain does not run it, and is charged
                    // in full whenever it does. Every other variant is
                    // unconditional, and a zero there is a variant nobody will
                    // notice is wrong.
                    let conditional = (matches!(
                        b,
                        WaveBuffer::QkvProjOperand
                            | WaveBuffer::QkvProjAccum
                            | WaveBuffer::OProjOperand
                            | WaveBuffer::OProjAccum
                    ) && !g.projection_accum_roundtrip)
                        || (matches!(
                            b,
                            WaveBuffer::GateSplit
                                | WaveBuffer::GateSigmoid
                                | WaveBuffer::GatedContext
                        ) && !g.gated_qkv)
                        || (matches!(b, WaveBuffer::QRotaryPermute | WaveBuffer::KRotaryPermute)
                            && !g.partial_rotary);
                    let s = b.shape(&g, rows);
                    if conditional {
                        assert_eq!(b.bytes(&g, rows), 0, "{b:?} priced while disabled");
                        continue;
                    }
                    assert!(b.bytes(&g, rows) > 0, "{b:?} sized zero at {rows} rows");
                    assert!(s.rows > 0 && s.cols > 0, "{b:?} has an empty shape");
                    let _ = b.phase();
                }
            }
        }
    }

    /// The float-session attention chain, pinned against the measured carve
    /// sizes of the 0.8B census (a C8 ×10-context wave, 5190 rows) rather
    /// than against itself. That census is the one that caught the span
    /// exhaustion: the gate's projection width, the two rotary-permute
    /// gathers, the gate split/sigmoid/apply, and `o_proj`'s round trip were
    /// all carved and none were priced.
    #[test]
    fn the_projection_round_trip_prices_the_measured_carves() {
        let g = float_projection();
        let rows = 5190;
        assert_eq!(
            WaveBuffer::QkvProjOperand.bytes(&g, rows),
            3 * 21_258_240,
            "the three hidden-wide F32 upcasts"
        );
        assert_eq!(
            WaveBuffer::QkvProjAccum.bytes(&g, rows),
            85_032_960 + 10_629_120 + 10_629_120,
            "the F32 [q|gate], K and V projection results"
        );
        assert_eq!(
            WaveBuffer::QkvProjection.bytes(&g, rows),
            42_516_480 + 5_314_560 + 5_314_560,
            "the act-dtype casts back down, gate width included"
        );
        assert_eq!(
            WaveBuffer::GateSplit.bytes(&g, rows),
            21_258_240,
            "the gate's contiguous copy off the strided narrow"
        );
        assert_eq!(WaveBuffer::GateSigmoid.bytes(&g, rows), 21_258_240);
        assert_eq!(WaveBuffer::GatedContext.bytes(&g, rows), 21_258_240);
        assert_eq!(
            WaveBuffer::QRotaryPermute.bytes(&g, rows),
            21_258_240,
            "Q re-ordered for the full-width rotary kernels"
        );
        assert_eq!(WaveBuffer::KRotaryPermute.bytes(&g, rows), 5_314_560);
        assert_eq!(
            WaveBuffer::OProjOperand.bytes(&g, rows),
            42_516_480,
            "o_proj's F32 operand upcast"
        );
        assert_eq!(
            WaveBuffer::OProjAccum.bytes(&g, rows),
            21_258_240,
            "o_proj's F32 result before the cast back"
        );
        // And a geometry with none of the flags is charged nothing for any of
        // them, so no packed model's span moves.
        let packed = ModelGeometry {
            projection_accum_roundtrip: false,
            gated_qkv: false,
            partial_rotary: false,
            ..g
        };
        for b in [
            WaveBuffer::QkvProjOperand,
            WaveBuffer::QkvProjAccum,
            WaveBuffer::GateSplit,
            WaveBuffer::GateSigmoid,
            WaveBuffer::GatedContext,
            WaveBuffer::QRotaryPermute,
            WaveBuffer::KRotaryPermute,
            WaveBuffer::OProjOperand,
            WaveBuffer::OProjAccum,
        ] {
            assert_eq!(b.bytes(&packed, rows), 0, "{b:?} priced while disabled");
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

    /// Per-row cost of the attention chain a **prefill** group runs, as
    /// `KV_WAVE_CENSUS=1` measured it on Qwen3-30B-A3B: twelve carves, and every
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

    /// The FFN chain on the **threaded expert pipeline**, which is the wider of
    /// the two dispatch paths — the GPU-native path scatters straight out of the
    /// down GEMM and never materialises [`WaveBuffer::DownGemm`] or
    /// [`WaveBuffer::DownCast`] on the span.
    ///
    /// Excludes the routing tables, which are bounded rather than measured
    /// (see [`ROUTING_U32_PER_ASSIGNMENT`]) — they came to 3.5 u32 per
    /// assignment against the 8 charged here.
    const MEASURED_FFN_PIPELINE_PER_ROW: usize = 183616;

    /// The plan must cover every chain that can run in its phase, because a
    /// mixed wave runs more than one of them inside a single generation.
    ///
    /// This is the assertion the whole module exists to make true, and it was
    /// false by 1.8x on attention and by 2x on the accumulate dtype until the
    /// census measured it. Stated against constants read off a real run rather
    /// than against the plan's own arithmetic — a test that recomputed the
    /// declaration would have passed throughout.
    #[test]
    fn the_plan_covers_every_measured_chain() {
        let plan = WavePlan::new(moe());
        for rows in [1usize, 20, 124, 744, 3936] {
            let attn = plan.phase_bytes(LayerPhase::Attention, rows);
            for (name, rate) in [
                ("prefill", MEASURED_ATTN_PREFILL_PER_ROW),
                ("decode", MEASURED_ATTN_DECODE_PER_ROW),
            ] {
                assert!(
                    attn >= rate * rows,
                    "attention at {rows} rows prices {attn} B but the measured \
                     {name} chain takes {} B\n{}",
                    rate * rows,
                    plan.describe(rows)
                );
            }
            let ffn = plan.phase_bytes(LayerPhase::Ffn, rows);
            assert!(
                ffn >= MEASURED_FFN_PIPELINE_PER_ROW * rows,
                "FFN at {rows} rows prices {ffn} B but the measured pipeline \
                 chain takes {} B\n{}",
                MEASURED_FFN_PIPELINE_PER_ROW * rows,
                plan.describe(rows)
            );
        }
    }

    /// What the union costs over the widest single chain, recorded so the
    /// over-bound is a number someone chose rather than one nobody noticed.
    ///
    /// A pure-prefill wave pays for `DecodeContext` and `OProjOutput` it never
    /// allocates; a pure-decode wave pays for four reshape copies that `seq == 1`
    /// makes free. Both are the price of the plan being handed a total row count
    /// instead of a per-group split, and closing it means passing the split.
    ///
    /// The third contribution is [`WaveBuffer::AttnNorm`] priced dense against a
    /// census taken on an int8 run, where it was q8a128: 1792 B/row of the
    /// 58,624 the chain measured, or 3.1 points of the margin below. That one is
    /// not closable by passing more context — the two encodings are alternatives
    /// and the plan has to bound both.
    #[test]
    fn the_attention_union_costs_a_recorded_margin() {
        let plan = WavePlan::new(moe());
        let rows = 1000;
        let priced = plan.phase_bytes(LayerPhase::Attention, rows);
        let widest = MEASURED_ATTN_PREFILL_PER_ROW * rows;
        let margin = (priced as f64 / widest as f64 - 1.0) * 100.0;
        println!("attention union over the prefill chain: {margin:.1}%");
        assert!(
            (17.0..19.0).contains(&margin),
            "the union's margin over the widest chain moved to {margin:.1}% — \
             either a buffer was added to one chain only, or the shapes changed"
        );
    }

    #[test]
    fn the_ffn_phase_dominates_a_moe_layer() {
        let plan = WavePlan::new(moe());
        assert!(
            plan.phase_bytes(LayerPhase::Ffn, 64) > plan.phase_bytes(LayerPhase::Attention, 64),
            "expert replication should make the FFN the sizing phase:\n{}",
            plan.describe(64)
        );
    }

    /// A wave costs the larger phase, never the sum: the two generations do not
    /// overlap, so summing would price twice what a half holds at once.
    #[test]
    fn a_wave_costs_the_larger_phase_not_the_sum() {
        let plan = WavePlan::new(moe());
        let sum =
            plan.phase_bytes(LayerPhase::Attention, 64) + plan.phase_bytes(LayerPhase::Ffn, 64);
        assert!(plan.wave_bytes(64) < sum);
        assert!(plan.wave_bytes(64) >= plan.phase_bytes(LayerPhase::Ffn, 64));
    }

    /// The bisection in `max_rows_within` is only valid if cost never decreases
    /// with width. Assert the property it relies on rather than trusting it.
    #[test]
    fn wave_cost_is_non_decreasing_in_rows() {
        for g in [moe(), dense()] {
            let plan = WavePlan::new(g);
            let mut prev = 0;
            for rows in 1..600 {
                let cost = plan.wave_bytes(rows);
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
                let rows = plan.max_rows_within(budget);
                assert!(rows > 0, "budget {budget} should fit at least one row");
                assert!(
                    plan.fits(rows, budget),
                    "{rows} rows must fit in {budget} B"
                );
                assert!(
                    !plan.fits(rows + 1, budget),
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
        let widest = plan.max_rows_within(budget);
        assert!(plan.ensure_fits(widest, budget).is_ok());
        assert!(plan.ensure_fits(widest + 1, budget).is_err());
    }

    /// An over-budget refusal has to say enough to act on: how wide the wave
    /// was, how much it overran, and what would have fit.
    #[test]
    fn the_over_budget_error_names_the_width_and_the_overage() {
        let plan = WavePlan::new(moe());
        let budget = 8 * MIB;
        let rows = plan.max_rows_within(budget) + 64;
        let err = plan
            .ensure_fits(rows, budget)
            .expect_err("must refuse")
            .to_string();
        assert!(err.contains("wave over budget"), "{err}");
        assert!(err.contains(&rows.to_string()), "names the width: {err}");
        assert!(err.contains("over by"), "names the overage: {err}");
        assert!(err.contains("widest wave"), "names what would fit: {err}");
    }

    /// A budget too small to price a single token is a misconfiguration, and
    /// must be reported as zero rather than rounded up to one — admitting a
    /// wave the span cannot hold is the failure the gate exists to prevent.
    #[test]
    fn a_budget_below_one_row_admits_nothing() {
        let plan = WavePlan::new(moe());
        let one_row = plan.wave_bytes(1);
        assert_eq!(plan.max_rows_within(one_row - 1), 0);
        assert_eq!(plan.max_rows_within(one_row), 1);
    }

    /// Halving the budget must roughly halve the admitted width — the property
    /// that makes the half size a usable throughput/footprint dial.
    #[test]
    fn the_admitted_width_tracks_the_budget() {
        let plan = WavePlan::new(moe());
        let wide = plan.max_rows_within(64 * MIB);
        let narrow = plan.max_rows_within(32 * MIB);
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
            single.max_rows_within(64 * MIB) > routed.max_rows_within(64 * MIB),
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
        assert!(bf16_accum.max_rows_within(64 * MIB) > f32_accum.max_rows_within(64 * MIB));
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
        let rows = plan.max_rows_within(WAVE_FFN_BYTES);
        println!(
            "FFN span {} B admits {rows} rows; one row costs {} B",
            WAVE_FFN_BYTES,
            plan.phase_bytes(LayerPhase::Ffn, 1),
        );
        assert!(
            rows > 0,
            "a span that cannot price one row would stall every wave"
        );
        // Monotonic in the budget: halving the span must not admit more rows.
        assert!(plan.max_rows_within(WAVE_FFN_BYTES / 2) <= rows);
    }
}
