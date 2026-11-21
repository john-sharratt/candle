//! Checks that the plan covers every chain it has to, and charges each group only
//! for its own rows.
//!
//! A wave of prefill tokens must cover the prefill chain at its width, and a
//! decode wave must cover the decode chain at its width; neither is asked to cover
//! the other's. A pure-prefill or pure-decode wave pays nothing for the other
//! chain, while a mixed wave pays for both at their own widths, because one
//! attention generation holds both at once.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_fixtures::tests::{
        moe, prefill_headers, MEASURED_ATTN_DECODE_PER_ROW, MEASURED_ATTN_PREFILL_PER_ROW,
        MEASURED_FFN_PIPELINE_PER_ROW,
    };

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
            MEASURED_ATTN_PREFILL_PER_ROW * rows + prefill_headers(1),
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
                prefill_spans: 1,
                ..WaveWidth::default()
            },
        );
        assert_eq!(mixed, prefill_only + decode_only);
    }
}
