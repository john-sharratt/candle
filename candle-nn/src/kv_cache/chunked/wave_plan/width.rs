//! Describes how wide a wave is, in the three units its buffers scale with.
//!
//! A single row count cannot price a wave. A mixed wave runs two groups whose
//! buffers each scale with their own rows, and the forward phase scales with the
//! scored rows, one per sequence in the usual case. The width also carries the
//! multi-row spans a DeltaNet mixer runs through its prefill kernels, which set
//! the span tables and the scan's transients, and the sparse selection and prefill
//! stage, which depend on depth as well as width. `ffn_work_dtype` is the one
//! definition of the dtype the FFN carries its intermediates in.

use super::*;

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
    /// Prefill-group sequences contributing **more than one row** — the spans
    /// a DeltaNet mixer runs through its prefill kernels rather than its
    /// decode ones (a verifying block rides the prefill group, so it counts).
    ///
    /// Two costs hang on it and neither is a row count: the mixer's span
    /// tables hold an entry per such span, and the prefill scan's transients
    /// exist only when there is at least one — though they are then sized by
    /// the whole wave.
    pub prefill_spans: usize,
    /// Scored rows an accept walk selects **on the head's span** after this
    /// wave — [`WaveBuffer::AcceptRows`]. The scored rows when the caller reads
    /// the logits in place, zero when it owns them first (or runs no walk), so
    /// a wave reserves the walk's selections only when they will land there.
    pub accept_rows: usize,
    /// Rows the sampler holds to a grammar's bitmap after this wave —
    /// [`WaveBuffer::StencilBitmaps`]. The caller counts them before the forward:
    /// a row's mask is decided by then.
    pub stencil_rows: usize,
    /// Bytes one attention layer's **sparse selection** carves on its phase:
    /// the index keys and queries, the score buffer, the scoring products, the
    /// selection table and the per-span tables — [`WaveBuffer::QsaSelection`].
    ///
    /// Not a function of the wave's width alone, which is why the model states
    /// it rather than the geometry: the score buffer and the scoring products
    /// are as wide as the candidate blocks the deepest row can see, so they
    /// grow with *context depth*. A model whose attention selects nothing, or a
    /// wave whose rows are all within the dense budget, states zero for all of
    /// it but the index keys every attention layer appends.
    pub qsa_bytes: usize,
    /// What one attention layer's prefill launch carves beside its context —
    /// [`WaveBuffer::PrefillKvStage`]: the pre-staged K/V, the online-softmax
    /// carry and the resume table ([`PrefillKvStageLayout`] over the prefill
    /// group, at the geometry's heads).
    ///
    /// Not a function of the width: a staged sequence stages up to one key
    /// chunk of its context, so the stage follows the depth of the wave's bulk
    /// chunks until that bound. [`PrefillKvStageLayout::NONE`] for a wave with
    /// no prefill sequence of
    /// [`PREFILL_KV_STAGE_MIN_Q_LEN`](super::prefill_kv_stage::PREFILL_KV_STAGE_MIN_Q_LEN)
    /// rows.
    pub kv_stage: PrefillKvStageLayout,
    /// Later rows' routers a MoE layer also applies to its input, for the expert
    /// cache's look-ahead votes: each is `n_experts` more F32 columns of
    /// [`WaveBuffer::LookAheadLogits`]. Zero on a stack that votes on nothing.
    pub look_ahead_hops: usize,
    /// Rows whose ids name an image's rows rather than the vocabulary — the
    /// wave uploads each one's values as [`WaveBuffer::ImageRows`]. The caller
    /// counts them before the forward, from the wave's own ids; zero on a wave
    /// with no image in it.
    pub image_rows: usize,
}

impl WaveWidth {
    /// Every token in the wave, whichever group contributed it.
    pub const fn rows(&self) -> usize {
        self.prefill_rows + self.decode_rows
    }

    /// A wave that is all prefill: `rows` tokens over `sequences` sequences,
    /// none of them verifying, so the head scores one row each.
    ///
    /// The per-sequence lengths are not known here, so the multi-row spans are
    /// bounded: at most every sequence, and at most one per two rows.
    pub const fn prefill(rows: usize, sequences: usize) -> Self {
        let pairs = rows / 2;
        Self {
            prefill_rows: rows,
            decode_rows: 0,
            scored_rows: sequences,
            prefill_spans: if sequences < pairs { sequences } else { pairs },
            accept_rows: 0,
            stencil_rows: 0,
            qsa_bytes: 0,
            kv_stage: PrefillKvStageLayout::NONE,
            look_ahead_hops: 0,
            image_rows: 0,
        }
    }

    /// A wave that is all decode: one row per sequence, every one of them
    /// scored.
    pub const fn decode(sequences: usize) -> Self {
        Self {
            prefill_rows: 0,
            decode_rows: sequences,
            scored_rows: sequences,
            prefill_spans: 0,
            accept_rows: 0,
            stencil_rows: 0,
            qsa_bytes: 0,
            kv_stage: PrefillKvStageLayout::NONE,
            look_ahead_hops: 0,
            image_rows: 0,
        }
    }

    /// `rows` more prefill tokens on top of this wave, from **one more
    /// sequence** — a multi-row span once there are two of them.
    ///
    /// **One sequence, however many rows.** The name says so because reading it
    /// as "add these rows" is wrong in a way nothing catches: spans are a cost
    /// of their own (the mixer's span tables hold an entry per span, and the
    /// prefill scan's transients turn on with the first), so `tier_bytes` is
    /// superlinear in them. A caller that spreads the rows over several
    /// sequences and prices them through here is told the cost of one span and
    /// handed the bill for many — which is how a wave gets composed wider than
    /// the tier the placement will give it, refused, and re-formed unchanged.
    ///
    /// Use [`Self::prefill`] wherever the sequence count is known and more than
    /// one.
    pub const fn with_one_more_sequence(self, rows: usize) -> Self {
        Self {
            prefill_rows: self.prefill_rows + rows,
            prefill_spans: self.prefill_spans + if rows > 1 { 1 } else { 0 },
            ..self
        }
    }
}

/// Rows in one chunk of the DeltaNet prefill scan — the width of its
/// intra-chunk `kq` transient ([`WaveBuffer::DeltaNetLiveScanKq`]). The kernel's
/// own constant is `candle_kernels::delta_net::DELTA_NET_PREFILL_CHUNK`, out of
/// reach of a build without CUDA; a CUDA test holds the two equal.
pub const DELTA_NET_SCAN_CHUNK: usize = 64;

/// Bytes of one paged-attention `SlotHeader` — `sizeof(SlotHeader)` in
/// `candle-kernels/src/paged-decode/slot_types.cuh`. A prefill launch carves one
/// per sequence from its attention span ([`WaveBuffer::PrefillSlotHeaders`]).
pub const SLOT_HEADER_BYTES: usize = 32;

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
