//! Checks the prefill and joint width bounds against the tier and the KV they
//! share.
//!
//! The prefill bound covers every sequence but never more than one multi-row
//! span per two rows, and it is the exact inverse of the cost. A standing head is
//! charged against the bound, so the width does not ignore it. The joint bound
//! pays for the tier and the KV from one budget, admitting exactly the rows whose
//! tier plus whole-region KV fits, and a budget below one row prices nothing.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use crate::kv_cache::WAVE_SPAN_BYTES;

    use super::super::tests_fixtures::tests::moe;

    /// The prefill bound on multi-row spans: every sequence, but never more
    /// than one per two rows — a one-row prefill is a decode-kernel span.
    #[test]
    fn a_prefill_width_bounds_its_multi_row_spans() {
        assert_eq!(WaveWidth::prefill(2100, 4).prefill_spans, 4);
        assert_eq!(WaveWidth::prefill(1, 1).prefill_spans, 0);
        assert_eq!(WaveWidth::prefill(5, 4).prefill_spans, 2);
        assert_eq!(
            WaveWidth::decode(3).with_one_more_sequence(1).prefill_spans,
            0
        );
        assert_eq!(
            WaveWidth::decode(3).with_one_more_sequence(7).prefill_spans,
            1
        );
    }

    /// The bound is the inverse of the cost, and it must not overshoot: the
    /// widest accepted width fits, and one row more does not.
    #[test]
    fn max_prefill_rows_for_tier_is_exact_on_the_staircase() {
        let p = WavePlan::new(moe());
        let budget = WAVE_SPAN_BYTES;
        let head = WaveWidth::prefill(0, 1);
        let scratch = 4096;
        let rows = p.max_prefill_rows_for_tier(budget, head, scratch);
        assert!(rows > 0, "the guaranteed span must price at least one row");
        assert!(
            p.tier_bytes(head.with_one_more_sequence(rows), scratch) <= budget,
            "the accepted width must fit"
        );
        assert!(
            p.tier_bytes(head.with_one_more_sequence(rows + 1), scratch) > budget,
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
        let empty = p.max_prefill_rows_for_tier(budget, WaveWidth::prefill(0, 1), 0);
        let loaded = p.max_prefill_rows_for_tier(budget, WaveWidth::decode(32), 0);
        assert!(
            loaded < empty,
            "32 decode rows must cost prefill width: {loaded} vs {empty}"
        );
    }

    /// **The tier and the KV share the budget.** With no KV the joint bound is
    /// the tier bound; with KV it admits fewer rows, and exactly the most whose
    /// tier plus whole-region KV fits — one row more does not.
    #[test]
    fn the_joint_bound_pays_tier_and_kv_from_one_budget() {
        let p = WavePlan::new(moe());
        let budget = 4 * WAVE_SPAN_BYTES;
        let head = WaveWidth::prefill(0, 1);
        let scratch = 4096;
        assert_eq!(
            p.max_prefill_rows_for_tier_and_kv(budget, head, 0, scratch),
            p.max_prefill_rows_for_tier(budget, head, scratch),
        );

        // Qwen3-30B-A3B's KV per row: K and V, 4 KV heads × 128, BF16, 48 layers.
        let kv_row = 2 * 4 * 128 * 2 * 48;
        let rows = p.max_prefill_rows_for_tier_and_kv(budget, head, kv_row, scratch);
        assert!(rows < p.max_prefill_rows_for_tier(budget, head, scratch));
        let cost = |r: usize| {
            p.tier_bytes(head.with_one_more_sequence(r), scratch)
                + (r * kv_row).div_ceil(TARGET_ARENA_BYTES) * TARGET_ARENA_BYTES
        };
        assert!(cost(rows) <= budget, "the accepted width must fit");
        assert!(cost(rows + 1) > budget, "one row more must not fit");
    }

    /// A budget too small for a single row answers 0, which callers must treat
    /// as a configuration error rather than as an empty wave.
    #[test]
    fn a_budget_below_one_row_prices_nothing() {
        let p = WavePlan::new(moe());
        assert_eq!(p.max_prefill_rows_for_tier(1, WaveWidth::default(), 0), 0);
    }
}
