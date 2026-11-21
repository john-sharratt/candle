//! Pins the FFN chains to their measured generations, to the byte.
//!
//! The dense FFN is checked against four carves at 2,100 rows, and the expert
//! pipeline against two whole generations, one on a BF16 session and one on an F16
//! session, including the routing tables and their alignment pads. The shared
//! expert's carves are pinned on the 35B, which has one, and a look-ahead hop is
//! checked as F32 router logits. The census was taken on int8 sessions, which is
//! why the norm is priced packed.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_fixtures::tests::{
        dense, empty_head, gated_partial_rotary, moe, moe_35b, w, MEASURED_FFN_35B_BF16_2100,
        MEASURED_FFN_35B_F16_1070, MEASURED_FFN_BF16_4960, MEASURED_FFN_F16_9880,
    };

    /// Each look-ahead hop is one router's logits in F32, whatever the session's
    /// dtype: 100 rows × 128 experts × 4 bytes = 51,200 a hop, nothing without
    /// one, and the router's own logits (BF16 here) unchanged.
    #[test]
    fn a_look_ahead_hop_prices_one_routers_f32_logits() {
        let g = moe();
        let plain = WaveWidth::prefill(100, 1);
        let voting = |hops| WaveWidth {
            look_ahead_hops: hops,
            ..plain
        };
        assert_eq!(WaveBuffer::LookAheadLogits.bytes(&g, plain), 0);
        assert_eq!(WaveBuffer::LookAheadLogits.bytes(&g, voting(1)), 51_200);
        assert_eq!(WaveBuffer::LookAheadLogits.bytes(&g, voting(5)), 256_000);
        assert_eq!(WaveBuffer::RouterLogits.bytes(&g, voting(5)), 25_600);
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

    /// **A shared expert's carves are priced — all nine.** Its down projection,
    /// the q8a128 operand that feeds it and the gated product built on it carve
    /// from the FFN generation like the rest; left out of the plan, they took
    /// the routed down projection's room and pushed it onto the CUDA pool.
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
            WaveBuffer::SharedDownOperand,
            WaveBuffer::SharedDown,
            WaveBuffer::SharedGateLogits,
            WaveBuffer::SharedGateSigmoid,
            WaveBuffer::SharedGatedOut,
            WaveBuffer::MoeSharedSum,
        ] {
            assert_eq!(b.bytes(&moe(), width), 0, "{b:?}");
            assert!(b.bytes(&moe_35b(), width) > 0, "{b:?}");
        }
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

    /// What the FFN span actually admits on the production model, recorded so a
    /// change to either the span or the geometry has to face the number.
    ///
    /// The compute-side ceiling (`MAX_PREFILL_TOKENS`) and this are independent,
    /// and the caller takes the narrower. Printing both is what tells you which
    /// one is binding — a question that has no answer from either constant alone.
    #[test]
    fn the_ffn_span_admits_a_recorded_width() {
        use super::super::super::wave_spans::WAVE_FFN_BYTES;
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
