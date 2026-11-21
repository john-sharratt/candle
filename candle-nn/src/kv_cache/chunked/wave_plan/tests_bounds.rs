//! Checks that the width bounds are exact, monotone, and refuse what they should.
//!
//! The bisection in `max_rows_within` relies on cost never falling as width
//! grows, so the tests assert that property directly rather than trusting it.
//! The bound is checked at its edge: the returned width fits and one more row
//! does not, and `ensure_fits` accepts exactly what the bound promises. A budget
//! too small for one row must answer zero rather than round up, because admitting
//! a wave the span cannot hold is the failure the gate exists to prevent.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_fixtures::tests::{dense, empty_head, moe, w, MIB};

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
}
