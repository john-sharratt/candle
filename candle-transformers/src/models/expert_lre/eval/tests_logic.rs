//! Deterministic tests of the replay and metric logic, which need no fixture.
//!
//! The tests check that passes are grouped by config and pass, that precision and
//! coverage count hits against the next layer's set and its miss set, and that a
//! perfectly periodic trace is predicted exactly once the matrix is warm. Only
//! the first few passes of that trace are cold, so the rest of its transitions
//! are scored as perfect.

use super::*;

// ── Deterministic tests of the replay/metric logic (no fixture needed) ──

#[cfg(test)]
pub(super) mod logic_tests {
    use super::*;

    pub(in super::super) fn rec(
        config: u16,
        pass: u32,
        layer: u16,
        experts: &[u32],
    ) -> RoutingRecord {
        RoutingRecord {
            config,
            pass,
            layer,
            experts: experts.to_vec(),
            mass: vec![1.0; experts.len()],
        }
    }

    #[test]
    fn groups_passes_by_config_and_pass() {
        let recs = vec![
            rec(0, 0, 0, &[1]),
            rec(0, 0, 1, &[2]),
            rec(0, 1, 0, &[3]),
            rec(1, 0, 0, &[4]),
        ];
        let t = Trace::from_records(&recs);
        assert_eq!(t.passes.len(), 3);
        assert_eq!(t.passes[0].layers.len(), 2);
        assert_eq!(t.passes[1].layers.len(), 1);
        assert_eq!(t.num_layers, 2);
        assert_eq!(t.num_experts, 5);
    }

    #[test]
    fn metrics_count_precision_and_coverage() {
        let mut m = Metrics::default();
        // current = {0}, next = {1,2}; predicted = {1,3}.
        // pred_total=2, pred_hits=1 (1∈next), miss={1,2}, miss_covered=1 (1).
        m.add(&[1, 3], &[0], &[1, 2]);
        assert_eq!(m.pred_total, 2);
        assert_eq!(m.pred_hits, 1);
        assert_eq!(m.miss_total, 2);
        assert_eq!(m.miss_covered, 1);
        assert_eq!(m.top1_hits, 1); // predicted[0]=1 ∈ next
        assert!((m.precision() - 0.5).abs() < 1e-9);
        assert!((m.coverage() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn perfectly_periodic_trace_is_learned() {
        // Identical routing every pass: 0:{0} -> 1:{5} -> 2:{9}.  Once the
        // matrix is warm (64 observations = 32 passes), every transition is
        // predicted exactly.  Use enough passes that the warmup is negligible.
        let mut many = Vec::new();
        for p in 0..700u32 {
            many.push(rec(0, p, 0, &[0]));
            many.push(rec(0, p, 1, &[5]));
            many.push(rec(0, p, 2, &[9]));
        }
        let trace = Trace::from_records(&many);
        let mut mat = TransitionMatrix::new(trace.num_layers, trace.num_experts);
        let (all, _) = replay(&trace, &mut mat, 1, 16, 0);
        // Cold start is ~32/700 of transitions; the rest are perfect.
        assert!(all.coverage() > 0.9, "coverage was {}", all.coverage());
        assert!(all.top1() > 0.9, "top1 was {}", all.top1());
    }
}
