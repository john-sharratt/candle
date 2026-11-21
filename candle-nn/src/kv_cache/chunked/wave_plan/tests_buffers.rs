//! Checks each buffer variant's phase and size across real and degenerate
//! geometries.
//!
//! Byte arithmetic is pinned against hand-computed values, and a q8a128
//! super-block rounds a partial group up to a whole block. Every variant must name
//! a phase and a non-zero size for a full geometry, so a zero where none is
//! expected is a defect rather than a width the case happened not to supply. The
//! partition tests confirm that the phases and the chains each cover every buffer
//! exactly once, and that a buffer's phase is its chain's phase.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::prefill_kv_stage::{PrefillLaunchBounds, PREFILL_KV_STAGE_MIN_Q_LEN};

    use super::*;

    use super::super::tests_fixtures::tests::{dense, gated_partial_rotary, moe};

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
                        // The prefill context's quantize, which only an int8
                        // `o_proj` makes.
                        || (matches!(b, WaveBuffer::OProjOperand) && !g.packed_norm)
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
                                | WaveBuffer::SharedDownOperand
                                | WaveBuffer::SharedDown
                                | WaveBuffer::SharedGateLogits
                                | WaveBuffer::SharedGateSigmoid
                                | WaveBuffer::SharedGatedOut
                                | WaveBuffer::MoeSharedSum
                        ) && g.shared_expert.is_none())
                        // A float head carries F32 working copies of its norm
                        // and its logits; a packed one does not.
                        || (matches!(b, WaveBuffer::HeadNormF32) && g.packed_head)
                        || (matches!(b, WaveBuffer::HeadLogitsF32)
                            && (g.packed_head || g.act_dtype == DType::F32))
                        || (b.chain() == Chain::DeltaNet && g.delta_net.is_none())
                        || (b.chain() == Chain::Ffn && !g.is_moe())
                        || (b.chain() == Chain::DenseFfn && g.is_moe())
                        // The Gated Residual pre-mix, on a stack that has one.
                        || (matches!(
                            b.chain(),
                            Chain::HyperAttn | Chain::HyperFfn | Chain::HyperHead
                        ) && g.hyper.is_none())
                        // A float head under the pre-mix: its norm is the mix.
                        || (matches!(b, WaveBuffer::HeadNorm | WaveBuffer::HeadNormF32)
                            && g.hyper.is_some()
                            && !g.packed_head)
                        // The stacked projection: a Gated Residual stack with
                        // DeltaNet layers only — where β and α ride it instead
                        // of their own.
                        || (matches!(
                            b,
                            WaveBuffer::DeltaNetLiveOperand
                                | WaveBuffer::DeltaNetLiveProjection
                                | WaveBuffer::DeltaNetLiveSplit
                        ) && (g.hyper.is_none() || g.delta_net.is_none()))
                        // The separate `[q|k|v]` and `z` projections, on a
                        // single-stream stack; their operands only where int8.
                        || (matches!(
                            b,
                            WaveBuffer::DeltaNetQkvProj | WaveBuffer::DeltaNetZProj
                        ) && (g.hyper.is_some() || g.delta_net.is_none()))
                        || (matches!(
                            b,
                            WaveBuffer::DeltaNetQkvOperand | WaveBuffer::DeltaNetZOperand
                        ) && (g.hyper.is_some() || g.delta_net.is_none() || !g.packed_norm))
                        || (matches!(
                            b,
                            WaveBuffer::DeltaNetBetaProj | WaveBuffer::DeltaNetAlphaProj
                        ) && g.hyper.is_some())
                        // DeltaNet's F32 upcasts are no-ops on an F32 session.
                        || (matches!(
                            b,
                            WaveBuffer::DeltaNetBetaOperand | WaveBuffer::DeltaNetAlphaOperand
                        ) && g.act_dtype == DType::F32)
                        // The draft head's assembly, on a stack without a head:
                        // the multi-stream head's, and the single-stream one's.
                        || (((b.chain() == Chain::HeadPass && !b.is_mtp())
                            || matches!(
                                b,
                                WaveBuffer::HeadEmbedNorm
                                    | WaveBuffer::HeadEmbedOperand
                                    | WaveBuffer::HeadEmbedProj
                            ))
                            && !g.hyper.is_some_and(|h| h.draft_head))
                        || (b.is_mtp() && !g.mtp_head)
                        // The PLE block, on a stack without one.
                        || (b.chain() == Chain::Ple && !g.hyper.is_some_and(|h| h.ple))
                        // The wide residual, on a single-stream stack; the
                        // rows' embeddings, on a stack without a draft head.
                        || (matches!(b, WaveBuffer::HyperResidual) && g.hyper.is_none())
                        // Glue rides the single-stream stacks only.
                        || (matches!(b, WaveBuffer::GlueWriteTables) && g.hyper.is_some())
                        // The decode table, on a stack with DeltaNet layers.
                        || (matches!(b, WaveBuffer::DeltaNetDecodeTable)
                            && g.delta_net.is_none())
                        // The single-stream residual; a multi-stream stack's is
                        // `HyperResidual`.
                        || (matches!(b, WaveBuffer::ForwardResidual) && g.hyper.is_some())
                        || (matches!(b, WaveBuffer::HeadRowEmbeds)
                            && !g.hyper.is_some_and(|h| h.draft_head));
                    // Every unit non-zero, so a buffer sized by any of the three
                    // is exercised and a zero is a defect rather than a width
                    // this case happened not to supply.
                    let width = WaveWidth {
                        prefill_rows: rows,
                        decode_rows: rows,
                        scored_rows: rows,
                        prefill_spans: rows,
                        accept_rows: rows,
                        stencil_rows: rows,
                        qsa_bytes: rows,
                        // A staged chunk of the line's rows, `rows` deep.
                        kv_stage: PrefillKvStageLayout::new(
                            &[PREFILL_KV_STAGE_MIN_Q_LEN],
                            &[rows],
                            g.n_head,
                            g.n_kv_head,
                            g.head_dim,
                            PrefillLaunchBounds::PRODUCTION,
                        ),
                        look_ahead_hops: rows,
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
}
