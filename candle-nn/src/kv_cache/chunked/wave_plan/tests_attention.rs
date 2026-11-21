//! Pins the attention chain's buffers to the census measured on real models,
//! carve for carve.
//!
//! Each test names the census or the geometry it reproduces: a gated 0.8B
//! chain at its peak generation, a stack with no per-head norms, a float session
//! with no fused projection, and a one-row wave that copies nothing out of its
//! fused row. Each assertion is a carve the census actually shows, so the plan's
//! arithmetic is checked against a measurement rather than against itself. The
//! sparse selection and the prefill stage are priced only where the model states
//! them, and their tests check that nothing is charged when nothing is stated.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::prefill_kv_stage::PrefillLaunchBounds;

    use super::*;

    use super::super::tests_fixtures::tests::{
        dense, gated_partial_rotary, hyper_hybrid, hyper_moe, moe, prefill_headers, w,
        MEASURED_ATTN_DECODE_ONE_ROW,
    };

    /// A layer's sparse selection is priced at what the model states, on the
    /// attention phase, and nothing when it states nothing.
    #[test]
    fn the_qsa_selection_is_priced_as_the_model_states_it() {
        let p = WavePlan::new(hyper_moe());
        let plain = WaveWidth::decode(4);
        let deep = WaveWidth {
            qsa_bytes: 3_000_000,
            ..plain
        };
        assert_eq!(WaveBuffer::QsaSelection.bytes(&hyper_moe(), plain), 0);
        assert_eq!(
            WaveBuffer::QsaSelection.bytes(&hyper_moe(), deep),
            3_000_000
        );
        // One more aligned carve at the chain's end: its bytes, plus at most
        // the alignment its start costs.
        let grew = p.chain_bytes(Chain::Attention, deep) - p.chain_bytes(Chain::Attention, plain);
        assert!(
            (3_000_000..3_000_000 + BUMP_ALIGNMENT).contains(&grew),
            "grew by {grew}"
        );
        assert_eq!(
            p.phase_bytes(LayerPhase::Ffn, deep),
            p.phase_bytes(LayerPhase::Ffn, plain),
            "the selection is the attention phase's alone"
        );
    }

    /// A wave with no multi-row span runs the decode kernels only: no span
    /// tables and none of the prefill scan's transients, while the recurrence
    /// output, the norm-gate and `w_out` are carved as on any wave.
    #[test]
    fn a_decode_wave_carves_no_prefill_scan() {
        let g = hyper_hybrid();
        let w = WaveWidth::decode(8);
        for b in [
            WaveBuffer::DeltaNetLiveSpanPtrs,
            WaveBuffer::DeltaNetLiveSpanExtents,
            WaveBuffer::DeltaNetLiveScanU,
            WaveBuffer::DeltaNetLiveScanW,
            WaveBuffer::DeltaNetLiveScanKq,
            WaveBuffer::DeltaNetLiveDecay,
        ] {
            assert_eq!(b.bytes(&g, w), 0, "{b:?}");
        }
        // 8 × 6144 × 4.
        assert_eq!(WaveBuffer::DeltaNetLiveScanOut.bytes(&g, w), 196_608);
        assert_eq!(WaveBuffer::DeltaNetLiveNormGate.bytes(&g, w), 196_608);
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
        // **The whole chain, to the byte**: the thirteen carves the census
        // itemises, plus the int8 `o_proj`'s quantized operand and result, which
        // inherit the context's arena (`MEASURED_ATTN_PREFILL_PER_ROW` records
        // the gate that ran out of span on them). `DecodeContext` is the decode
        // group's; a wave with no decode rows carves none, and priced at one
        // total row count it cost this span megabytes it could never spend.
        assert_eq!(
            WaveBuffer::OProjOperand.bytes(&g, rows),
            4_838_400,
            "q8a128 over the 2,048 attention columns of 2,100 rows"
        );
        assert_eq!(WaveBuffer::OProjOutput.bytes(&g, rows), 4_300_800);
        assert_eq!(
            WaveBuffer::PrefillSlotHeaders.bytes(&g, rows),
            rows.prefill_spans * SLOT_HEADER_BYTES + BUMP_ALIGNMENT,
            "a header per sequence, and room for a glue group's beside them"
        );
        assert_eq!(
            plan.chain_bytes(Chain::Attention, rows),
            88_435_200 + 4_838_400 + 4_300_800 + prefill_headers(rows.prefill_spans),
            "a pure-prefill wave must price its generation exactly"
        );
        assert_eq!(
            WaveBuffer::DecodeContext.bytes(&g, rows),
            0,
            "DecodeContext charged on a wave with no decode rows"
        );
        assert!(
            WaveBuffer::DecodeContext.bytes(&g, WaveWidth::decode(2100)) > 0,
            "DecodeContext must still be charged when there ARE decode rows"
        );
        // A float session's `o_proj` reads the context as it stands: no operand,
        // and the prefill group's result is not on the span.
        let float = ModelGeometry {
            packed_norm: false,
            ..g
        };
        assert_eq!(WaveBuffer::OProjOperand.bytes(&float, rows), 0);
        assert_eq!(WaveBuffer::OProjOutput.bytes(&float, rows), 0);

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
        // adds. The census (`KV_WAVE_CENSUS=labels`) itemises eight carves: the
        // norm, the fused projection, the three adds, the decode kernel's FP
        // context, the `o_proj` quantize of it and the `rows × hidden` result,
        // plus the pads the flat-grouped q8 buffers leave.
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
            613_888,
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

    /// The plan charges the launch's carve on the attention phase — the
    /// layout's bytes, built at the geometry's heads — and nothing for a wave
    /// that stages nothing. 32 query heads over 4 KV heads at head_dim 128:
    /// 392 B a position per KV head, 8 tokens a block, 512 blocks a group.
    #[test]
    fn the_stage_is_an_attention_buffer_priced_by_its_layout() {
        let g = hyper_moe();
        let plan = WavePlan::new(g);
        assert_eq!(WaveBuffer::PrefillKvStage.phase(), LayerPhase::Attention);
        assert_eq!(WaveBuffer::PrefillKvStage.chain(), Chain::Attention);
        let plain = w(2048);
        assert_eq!(WaveBuffer::PrefillKvStage.bytes(&g, plain), 0);
        let layout = |q: usize, off: usize| {
            PrefillKvStageLayout::new(
                &[q],
                &[off],
                g.n_head,
                g.n_kv_head,
                g.head_dim,
                PrefillLaunchBounds::PRODUCTION,
            )
        };
        // 2,048 rows at 8K: 10,240 positions inside one chunk, nothing carried —
        // K 5,242,880 + scales 327,680 + V 10,485,760.
        let shallow = WaveWidth {
            kv_stage: layout(2048, 8192),
            ..plain
        };
        assert_eq!(WaveBuffer::PrefillKvStage.bytes(&g, shallow), 16_056_320);
        // The same rows at 300K: one chunk of stage (51,380,224), one group of
        // carry (2,048 × 32 heads × 130 floats × 4 B = 34,078,720 — the
        // sequence is shorter than a group) and its resume table (256 blocks ×
        // 4 KV heads × 8 B = 8,192).
        let deep = WaveWidth {
            kv_stage: layout(2048, 300_000),
            ..plain
        };
        assert_eq!(WaveBuffer::PrefillKvStage.bytes(&g, deep), 85_467_136);
        let staged = shallow;
        let want = 16_056_320;
        // One more aligned carve on the attention chain: its bytes, plus at
        // most the alignment its start costs.
        let grew =
            plan.chain_bytes(Chain::Attention, staged) - plan.chain_bytes(Chain::Attention, plain);
        assert!(
            (want..want + BUMP_ALIGNMENT).contains(&grew),
            "the attention chain grew by {grew} B for a {want} B stage"
        );
    }
}
