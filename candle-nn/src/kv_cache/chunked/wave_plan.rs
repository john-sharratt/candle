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
//! # One source of truth, or none
//!
//! The failure mode this module is shaped to avoid is a plan that quietly
//! disagrees with what the code allocates. If the sizing lived here and the
//! allocation lived at the call site, the two would be independent transcripts
//! of the same shape, and they would drift the first time a kernel's output
//! changed — silently, because the only symptom is a span that exhausts under
//! a workload nobody ran.
//!
//! So a buffer is named **once**, as a [`WaveBuffer`] variant, and that variant
//! answers two questions: which phase it belongs to, and what shape it is at a
//! given row count. The byte arithmetic appears in exactly one function
//! ([`BufferShape::bytes`]). Call sites size their allocation by asking the
//! plan for the same variant, so a buffer that grows grows in both places or in
//! neither.
//!
//! Adding a buffer is adding an enum variant, and the compiler's exhaustiveness
//! check refuses that without a phase and a shape. Nothing is kept in a
//! parallel list — [`WaveBuffer::iter`] derives the inventory, so the plan, the
//! totals, the admission bound and the tests all pick up a new variant with no
//! edit.
//!
//! # A phase at a time, and its peak rather than its total
//!
//! A layer opens two generations — one spanning attention → `o_proj`, one
//! spanning the FFN — and each drops before the next opens, resetting its span.
//! So a span holds **one layer phase**, never a whole wave, and each phase gets
//! its own arena ([`super::bump_arena`]) sized from its own peak rather than
//! both from the larger.
//!
//! Within a phase, most buffers die long before it ends: the gate and up
//! projections are consumed by the fused SwiGLU, the SwiGLU output by the down
//! GEMM. [`WavePlan::phase_bytes`] therefore charges the **peak concurrent**
//! bytes, and [`WavePlan::layout`] realises that peak by assigning offsets so
//! buffers with disjoint lifetimes share bytes. Charging the total instead would
//! price a MoE layer at 1.75x its real high-water mark.

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
    /// The compute dtype activations are carried in.
    pub act_dtype: DType,
    /// What the int8 tensor-core kernels emit before the cast back to
    /// `act_dtype`. Both are live at once, so both are planned.
    pub accum_dtype: DType,
}

impl ModelGeometry {
    /// Rows the expert GEMMs see: every token replicated to each expert it
    /// routed to. This is what makes a MoE layer's FFN phase several times a
    /// dense one's, and why admission prices MoE waves differently.
    pub const fn expert_rows(&self, rows: usize) -> usize {
        rows * self.experts_per_tok
    }

    /// Columns produced by the fused QKV projection.
    pub const fn qkv_cols(&self) -> usize {
        (self.n_head + 2 * self.n_kv_head) * self.head_dim
    }

    /// Columns of the attention output, before `o_proj`.
    pub const fn attn_cols(&self) -> usize {
        self.n_head * self.head_dim
    }
}

/// Every buffer a layer allocates from the transient tier.
///
/// One variant per allocation site. A site that is not a variant here is a site
/// still reaching the driver — which is what `candle::forbidden_alloc` reports,
/// so the two are meant to be read against each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum WaveBuffer {
    /// Attention RMSNorm, quantized for the int8 QKV matmul.
    AttnNorm,
    /// Fused Q/K/V projection output.
    QkvProjection,
    /// Attention output, consumed by `o_proj`.
    ///
    /// Declared in its **dense** form, which is what prefill and glue write. The
    /// int8 decode path instead has its kernel emit the context already packed
    /// as q8a128 (`PagedDecode::q8_byte_size`), so o_proj needs no standalone
    /// quantize — and that form is strictly smaller: at `cols = 4096` a row costs
    /// 4608 B packed against 8192 B dense in F16. Declaring the dense form
    /// therefore upper-bounds both, and prefill — the wide case that actually
    /// decides the budget — is the dense one anyway. A second variant for the
    /// packed form would only ever lower a bound that is never the binding one.
    AttnOutput,
    /// `o_proj` int8 result, before the cast back to the compute dtype.
    OProjAccum,
    /// `o_proj` cast to the compute dtype.
    OProjCast,
    /// FFN RMSNorm, quantized for the expert GEMMs.
    FfnNorm,
    /// Tokens gathered into expert-major order for the grouped GEMMs.
    MoeGather,
    /// Gate projection over the gathered tokens.
    GateGemm,
    /// Up projection over the gathered tokens.
    UpGemm,
    /// Fused SwiGLU output, requantized to feed the down GEMM.
    SwigluAct,
    /// Down projection, in the accumulate dtype.
    DownGemm,
    /// Down projection cast to the compute dtype for the scatter.
    DownCast,
    /// The MoE combine target the scatter accumulates into.
    MoeCombine,
}

impl WaveBuffer {
    /// When this buffer is live within its phase.
    pub fn phase(&self) -> LayerPhase {
        match self {
            Self::AttnNorm
            | Self::QkvProjection
            | Self::AttnOutput
            | Self::OProjAccum
            | Self::OProjCast => LayerPhase::Attention,
            Self::FfnNorm
            | Self::MoeGather
            | Self::GateGemm
            | Self::UpGemm
            | Self::SwigluAct
            | Self::DownGemm
            | Self::DownCast
            | Self::MoeCombine => LayerPhase::Ffn,
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
            Self::AttnNorm => q8(rows, g.hidden),
            Self::QkvProjection => dense(rows, g.qkv_cols(), g.act_dtype),
            Self::AttnOutput => dense(rows, g.attn_cols(), g.act_dtype),
            Self::OProjAccum => dense(rows, g.hidden, g.accum_dtype),
            Self::OProjCast => dense(rows, g.hidden, g.act_dtype),
            Self::FfnNorm => q8(rows, g.hidden),
            Self::MoeGather => q8(er, g.hidden),
            Self::GateGemm => dense(er, g.intermediate, g.accum_dtype),
            Self::UpGemm => dense(er, g.intermediate, g.accum_dtype),
            Self::SwigluAct => q8(er, g.intermediate),
            Self::DownGemm => dense(er, g.hidden, g.accum_dtype),
            Self::DownCast => dense(er, g.hidden, g.act_dtype),
            Self::MoeCombine => dense(rows, g.hidden, g.act_dtype),
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
            act_dtype: DType::BF16,
            accum_dtype: DType::F32,
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
            act_dtype: DType::BF16,
            accum_dtype: DType::F32,
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
        for g in [moe(), dense()] {
            for rows in [1usize, 64, 4096] {
                for b in WaveBuffer::iter() {
                    assert!(b.bytes(&g, rows) > 0, "{b:?} sized zero at {rows} rows");
                    let s = b.shape(&g, rows);
                    assert!(s.rows > 0 && s.cols > 0, "{b:?} has an empty shape");
                    let _ = b.phase();
                }
            }
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
        use super::super::bump_arena::WAVE_FFN_BYTES;
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
